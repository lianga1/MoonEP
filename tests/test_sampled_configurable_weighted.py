"""Weighted K16/K32 forward and saved-plan backward on eight physical GPUs.

torchrun --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_configurable_weighted.py

The numerical expert is y_e = a_e * router_weight_e * x, with a_e in {1,2}.
This checks forward output, dX and individual router-weight gradients through
the public token communication APIs. It does not test GEMM, parameter-gradient
reduction, an optimizer, or performance; those have separate tests.
"""

import pytest
import torch

from moonep import Buffer, SampledPlannerConfig
from tests.test_sampled_configurable import (
    R, H, SEED, PLAN_FIELDS, _audit, _assert_all, _gather_cpu, _hidden,
    _log, _ready, _routes, eight_physical,
)


CASES = [(512, 16, 33), (1024, 32, 17)]


def _wait(result):
    if isinstance(result[-1], torch.cuda.Event):
        torch.cuda.current_stream().wait_event(result[-1])
    torch.cuda.synchronize()


def _compile_used(buffer, rank, label):
    """Compile only weighted-fresh/unweighted-reuse dispatch and both combines.

    All ranks complete JIT before any bounded native device barrier launches.
    There is no first-use compilation between one rank entering a collective
    kernel and another rank launching that kernel.
    """
    from moonep import (planning, inter_rank_sync, dispatch, combine,
                        dispatch_epilogue, combine_prologue)

    c = buffer._require_ctx()
    r, e, b, s, k, h, nvs, nvsp, stride, sms, device = (int(c[key]) for key in
        ("R", "E", "B", "S", "K", "H", "NvS", "NvS_padded", "meta_chunk_padded", "num_sms", "device"))
    _log(rank, label, "compile native planner and synchronization")
    planning._get_compiled(r, e, b, s, k, int(c["NvS_capacity"]), nvs,
        int(c["num_vblocks"]), stride, int(c["TPE_OFF"]), int(c["PLAN_OFF"]),
        int(c["BARRIER_OFF"]), int(c["TOPK0_OFF"]), int(c["ORDER_OFF"]),
        int(c["ORDER0_OFF"]), int(c["token_padding"]), sms)
    inter_rank_sync._get_compiled(r, stride, inter_rank_sync._num_threads_for_ranks(r), device)
    for with_weights, fresh in ((True, True), (False, False)):
        _log(rank, label, f"compile dispatch weights={with_weights}, fresh={fresh}")
        dispatch._get_compiled(h, r, s, k, e+b, nvs, nvsp, stride,
            int(c["SRC_INFO_OFF"]), sms, with_weights, fresh, device, False)
    for with_weights in (False, True):
        _log(rank, label, f"compile combine router-gradient output={with_weights}")
        combine._get_compiled(h, r, s, k, nvs, nvsp, stride, sms, with_weights, device, False)
    _log(rank, label, "compile epilogue/prologue; wait for every rank")
    dispatch_epilogue._get_compiled(h, nvs, int(c["num_sms_dedup"]), device, False)
    combine_prologue._get_compiled(h, r, nvs, int(c["num_sms_dedup"]), device, False)
    _ready()
    _log(rank, label, "all ranks compiled; begin weighted execution")


def _router_weights(rank, s, k, phase):
    pair = torch.arange(s*k, device="cuda", dtype=torch.int64).view(s, k)+rank*s*k
    # Binary fractions make each per-expert product exactly representable in
    # BF16 before the native FP32 combine sum. Phase also changes the weights.
    return ((pair+phase) % 2+1).float()/8


def _forward(buffer, topk, tpe, cfg, phase, asynchronous):
    c = buffer._require_ctx()
    rank, s, k = (int(c[name]) for name in ("rank", "S", "K"))
    x = _hidden(rank, s, phase)
    weights = _router_weights(rank, s, k, phase)
    result = buffer.dispatch(x, weights, topk, tpe, async_finish=asynchronous)
    _wait(result)
    dispatched, routed_weights, cu, plan = result[:4]
    assert routed_weights is not None
    routes, owners, offsets = _audit(buffer, topk, plan, cu, cfg)
    local = owners == rank
    pair_ids = torch.nonzero(local.flatten()).flatten()
    local_offsets = offsets[local].to(device="cuda", dtype=torch.long)
    expected_x = _gather_cpu(x).reshape(R*s, H)[pair_ids//k]
    expected_weights = _gather_cpu(weights).flatten()[pair_ids]
    _assert_all(torch.equal(dispatched[local_offsets].cpu(), expected_x), "weighted dispatch payload mismatch")
    _assert_all(torch.equal(routed_weights[local_offsets].cpu(), expected_weights), "per-pair routing weight mismatch")
    _assert_all(dispatched.data_ptr() != c["hidden_buf_local"].data_ptr(), "forward activation unexpectedly aliases scratch")
    _assert_all(routed_weights.data_ptr() != c["weights_buf_local"].data_ptr(), "routing weights unexpectedly alias scratch")
    factor = (routes[local] % 2+1).to(device="cuda", dtype=torch.float32)
    selected_x = dispatched[local_offsets].clone()
    selected_weights = routed_weights[local_offsets].clone()
    y_rows = torch.zeros_like(dispatched)
    y_rows[local_offsets] = (selected_x.float()*factor[:, None]*selected_weights[:, None]).to(torch.bfloat16)
    combined = buffer.combine(plan=plan, hidden_nvsh=y_rows, async_finish=asynchronous)
    _wait(combined)
    coefficient = ((topk % 2+1).float()*weights).sum(1, keepdim=True)
    expected_y = (x.float()*coefficient).to(torch.bfloat16)
    _assert_all(torch.equal(combined[0], expected_y), "weighted forward sum mismatch")
    assert combined[1] is None
    return dict(plan=plan.clone(), cu=cu.clone(), topk=topk.clone(),
                x=x.clone(), weights=weights.clone(), coefficient=coefficient.clone(),
                selected_x=selected_x, selected_weights=selected_weights,
                offsets=local_offsets, factor=factor, pair_ids=pair_ids)


def _backward(buffer, saved, cfg, version, asynchronous):
    c = buffer._require_ctx()
    rank, s, k = (int(c[name]) for name in ("rank", "S", "K"))
    plan = saved["plan"]
    snapshot = {name: getattr(plan, name).clone() for name in PLAN_FIELDS}
    dy = _hidden(rank, s, version)
    result = buffer.dispatch(dy, plan=plan, async_finish=asynchronous)
    _wait(result)
    scattered_dy, no_weights, no_cu, returned = result[:4]
    assert no_weights is None and no_cu is None and returned is plan
    _assert_all(all(torch.equal(getattr(plan, name), value) for name, value in snapshot.items()),
                "saved-plan dispatch overwrote forward metadata")
    _audit(buffer, saved["topk"], plan, saved["cu"], cfg, current=False)
    offsets, factor = saved["offsets"], saved["factor"]
    expected_dy = _gather_cpu(dy).reshape(R*s, H)[saved["pair_ids"]//k]
    _assert_all(torch.equal(scattered_dy[offsets].cpu(), expected_dy), "saved-plan dY dispatch mismatch")
    dx_rows = torch.zeros_like(scattered_dy)
    dx_rows[offsets] = (scattered_dy[offsets].float()*factor[:, None]
                       * saved["selected_weights"][:, None]).to(torch.bfloat16)
    dw_rows = torch.zeros(scattered_dy.shape[0], device="cuda", dtype=torch.float32)
    dw_rows[offsets] = (saved["selected_x"].float()*scattered_dy[offsets].float()).sum(1)*factor
    combined = buffer.combine(plan=plan, hidden_nvsh=dx_rows,
                              route_weights_nvs=dw_rows, async_finish=asynchronous)
    _wait(combined)
    expected_dx = (dy.float()*saved["coefficient"]).to(torch.bfloat16)
    expected_dw = ((saved["x"].float()*dy.float()).sum(1, keepdim=True)
                   * (saved["topk"] % 2+1).float())
    _assert_all(torch.equal(combined[0], expected_dx), "input gradient dX mismatch")
    _assert_all(combined[1] is not None and torch.equal(combined[1], expected_dw), "individual router-weight gradient mismatch")
    _assert_all(all(torch.equal(getattr(plan, name), value) for name, value in snapshot.items()),
                "saved-plan backward overwrote forward metadata")


@pytest.mark.parametrize("case", CASES, ids=[f"E{e}_K{k}_S{s}" for e, k, s in CASES])
def test_weighted_forward_and_saved_plan_backward(eight_physical, monkeypatch, case):
    rank = eight_physical
    e, k, s = case
    label = f"weighted_E{e}_K{k}_S{s}"
    cfg = SampledPlannerConfig(sample_size=128, groups_per_class=2, rounds=2, sample_seed=SEED)
    asynchronous = k == 32
    _log(rank, label, f"construct with default B=E/R; async={asynchronous}")
    buffer = Buffer(s, H, k, e, R, planning_backend="sampled_group", planner_config=cfg,
                    token_padding=128, num_sms=32, enable_pdl=False, explicitly_destroy=True)
    try:
        assert int(buffer._require_ctx()["B"]) == e//R
        _compile_used(buffer, rank, label)
        adapter = buffer._require_ctx()["sampled_group_planner"]
        run, calls = adapter.run, []
        def counted(*args, **kwargs):
            calls.append(1)
            return run(*args, **kwargs)
        monkeypatch.setattr(adapter, "run", counted)
        saved = []
        for phase in (0, 1):
            _log(rank, label, f"weighted forward phase={phase} ({'local' if phase == 0 else 'hot'})")
            topk, tpe = _routes(rank, e, k, s, phase)
            saved.append(_forward(buffer, topk, tpe, cfg, phase, asynchronous))
        _assert_all(len(calls) == 2, "fresh weighted dispatch did not call sampled planner exactly twice")
        _assert_all(not torch.equal(saved[0]["topk"], saved[1]["topk"]), "dynamic routes did not change")
        for index, item in enumerate(saved):
            _log(rank, label, f"backward saved phase={index} after both forward plans")
            _backward(buffer, item, cfg, index+11, asynchronous)
        _assert_all(len(calls) == 2, "saved-plan backward unexpectedly reran the planner")
        _ready()
        _log(rank, label, "PASS weighted forward, dX, routing gradients and saved-plan lifetime")
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
