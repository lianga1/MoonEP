"""Variable sampled-group settings on EIGHT actual physical GPUs.

torchrun --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_configurable.py

Every runtime case uses public dispatch/combine with identity expert outputs,
full global address/layout/load audits and saved-plan reuse after a new plan.
The E512/K16 case also replays one graph with three different runtime inputs.
This is numerical integration coverage, not a performance/model-training test.
"""

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import moonep.api as api
from moonep import Buffer, SampledPlannerConfig
from tests.affinity_utils import check_expert_rows, gather_raw, plan_metrics
from tests.kernel_test_utils import _dedup_group_map


R, H, SEED = 8, 256, 20261026
# E, K, requested global samples, groups/class, rounds, tokens/source, slots.
CASES = [
    (64, 1, 37, 1, 1, 17, 8),
    (128, 3, 64, 2, 2, 33, 16),
    (256, 4, 128, 3, 2, 65, 32),
    (352, 8, 256, 2, 4, 257, 8),
    (520, 7, 37, 1, 2, 65, 65),
    (512, 16, 512, 2, 4, 65, 64),
    (1024, 32, 128, 2, 2, 33, 128),
    (4096, 1, 256, 1, 1, 1, 512),
    (64, 1, 4096, 8, 1, 513, 8),
    (8, 8, 37, 2, 2, 17, 1),
]
PLAN_FIELDS = ("dst", "experts_to_copy", "zero_fill_ranges", "remote_stats",
               "dup_groups", "dup_loffs", "dup_counts")


def _name(case):
    e, k, d, g, rounds, s, b = case
    return f"E{e}_K{k}_D{d}_G{g}_L{rounds}_S{s}_B{b}"


def _log(rank, label, stage):
    if rank == 0:
        print(f"[sampled configurable] {label}: {stage}", flush=True)


def _ready():
    torch.cuda.synchronize()
    dist.barrier()


def _assert_all(ok, label):
    flag = torch.tensor([int(bool(ok))], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert int(flag.item()) == 1, label


@pytest.fixture
def eight_physical(dist_env):
    rank, world = dist_env
    if world != R:
        pytest.skip("sampled configurable runtime requires eight actual ranks")
    assert int(os.environ.get("LOCAL_WORLD_SIZE", 0)) == R
    assert torch.cuda.device_count() >= R
    torch.set_num_threads(1)
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    identities = [None]*R
    dist.all_gather_object(identities, (socket.gethostname(),
        str(getattr(prop, "uuid", torch.cuda.current_device()))))
    assert len({host for host, _ in identities}) == 1, "single-node NVLink test only"
    assert len(set(identities)) == R, "each rank must use a different physical GPU"
    return rank


def _compile_used(buffer, rank, label):
    """No native collective launches until all ranks finish every used JIT."""
    from moonep import (planning, inter_rank_sync, dispatch, combine,
                        dispatch_epilogue, combine_prologue)

    c = buffer._require_ctx()
    r, e, b, s, k, h, nvs, nvsp, stride, sms, device = (int(c[key]) for key in
        ("R", "E", "B", "S", "K", "H", "NvS", "NvS_padded", "meta_chunk_padded", "num_sms", "device"))
    _log(rank, label, "compile native planner")
    planning._get_compiled(r, e, b, s, k, int(c["NvS_capacity"]), nvs,
        int(c["num_vblocks"]), stride, int(c["TPE_OFF"]), int(c["PLAN_OFF"]),
        int(c["BARRIER_OFF"]), int(c["TOPK0_OFF"]), int(c["ORDER_OFF"]),
        int(c["ORDER0_OFF"]), int(c["token_padding"]), sms)
    _log(rank, label, "compile native sync")
    inter_rank_sync._get_compiled(r, stride, inter_rank_sync._num_threads_for_ranks(r), device)
    for fresh in (True, False):
        _log(rank, label, f"compile unweighted dispatch fresh={fresh}")
        dispatch._get_compiled(h, r, s, k, e+b, nvs, nvsp, stride,
            int(c["SRC_INFO_OFF"]), sms, False, fresh, device, False)
    _log(rank, label, "compile unweighted combine and prologue/epilogue")
    combine._get_compiled(h, r, s, k, nvs, nvsp, stride, sms, False, device, False)
    dispatch_epilogue._get_compiled(h, nvs, int(c["num_sms_dedup"]), device, False)
    combine_prologue._get_compiled(h, r, nvs, int(c["num_sms_dedup"]), device, False)
    _log(rank, label, "wait for every rank's JIT completion")
    _ready()
    _log(rank, label, "all ranks compiled; start actual GPU execution")


def _routes(rank, e, k, s, phase):
    epn = e//R
    token = torch.arange(s, device="cuda", dtype=torch.int64)[:, None]
    expert_slot = torch.arange(k, device="cuda", dtype=torch.int64)[None, :]
    if k > epn:
        # K may exceed the number of experts homed on one rank. Use unique
        # choices across home ranks; this includes E=K=8, one home expert/rank.
        x = (token*k+expert_slot+rank*epn+phase*3) % e
    elif phase % 2:
        # Even the B8/E352 case has at most K=8 active remote experts.
        x = (((phase+2) % R)*epn + expert_slot).expand(s, k)
    else:
        x = rank*epn + (token*k+expert_slot+phase*3) % epn
    x = x.to(torch.int32).contiguous()
    return x, torch.bincount(x.flatten().long(), minlength=e).int()


def _hidden(rank, s, version):
    # Four base-eight digits distinguish every source/token, and exact /8
    # values remain representable after multiplication by every K<=32.
    token = torch.arange(s, device="cuda", dtype=torch.int64)+rank*s+version*37
    columns = torch.arange(H, device="cuda", dtype=torch.int64) % 4
    return (((token[:, None] >> (columns[None, :]*3)) & 7).float()/8).to(torch.bfloat16).contiguous()


def _gather_cpu(x):
    pieces = [torch.empty_like(x) for _ in range(R)]
    dist.all_gather(pieces, x)
    return torch.stack(pieces).cpu()


def _audit(buffer, topk, plan, cu, cfg, *, current=True):
    c = buffer._require_ctx()
    e, k, s, b, rank, nvs = (int(c[key]) for key in ("E", "K", "S", "B", "rank", "NvS"))
    routes = _gather_cpu(topk).long()
    raw = gather_raw(plan).reshape(R, s, k)
    metrics = plan_metrics(raw, routes, nvs, e)
    assert metrics["loads"] == [s*k]*R
    assert max(metrics["active_replicas_per_rank"]) <= b
    check_expert_rows(plan, cu, routes, c)
    address = torch.where(raw < 0, -raw-1, raw).long()
    owners, offsets = address//nvs, address % nvs
    counts = torch.bincount((owners*e+routes).flatten(), minlength=R*e).view(R, e)
    copies = plan.experts_to_copy.cpu().long()
    all_copies = _gather_cpu(plan.experts_to_copy)
    assert bool((all_copies == copies).all()), "copy tables differ across ranks"
    groups = torch.full((e+b,), -1, dtype=torch.long)
    groups[rank*(e//R):(rank+1)*(e//R)] = torch.arange(rank*(e//R), (rank+1)*(e//R))
    groups[e:] = copies[rank]
    active = groups >= 0
    real = torch.zeros(e+b, dtype=torch.long)
    real[active] = counts[rank, groups[active]]
    padding = int(c["token_padding"])
    padded = (real+padding-1)//padding*padding
    expected_cu = padded.cumsum(0)
    starts = expected_cu-padded
    expected_zero = torch.stack((torch.where(padded > real, starts+real, 0), padded-real), dim=1)
    _assert_all(torch.equal(cu.cpu().long(), expected_cu), "cu padding/layout mismatch")
    _assert_all(torch.equal(plan.zero_fill_ranges.cpu().long(), expected_zero), "padding zero-fill ranges mismatch")
    epn = e//R
    incoming = int(((counts[rank] > 0) & (torch.arange(e)//epn != rank)).sum())
    outgoing = int(((copies >= 0) & (copies//epn == rank)).sum())
    _assert_all(plan.remote_stats.cpu().tolist() == [incoming, outgoing], "remote_stats mismatch")
    expected_dedup = {}
    for source in range(R):
        for token in range(s):
            ks = torch.nonzero(owners[source, token] == rank).flatten().tolist()
            if len(ks) > 1:
                rows = offsets[source, token, ks].tolist()
                expected_dedup[rows[0]] = tuple(sorted(rows[1:]))
    actual_dedup, dedup_errors = _dedup_group_map(plan)
    _assert_all(not dedup_errors and actual_dedup == expected_dedup, "fresh/saved dedup map differs from actual addresses")
    if current:
        adapter = c["sampled_group_planner"]
        _assert_all(int(adapter.status.item()) == 0, "unexpected native fallback")
        _assert_all(torch.equal(adapter.routes, topk), "stale runtime routes")
        assert adapter.config == cfg and buffer.planner_config == cfg
        assert adapter.sample_size == min(cfg.sample_size, R*s)
        assert adapter.groups_per_class == cfg.groups_per_class and adapter.rounds == cfg.rounds
        assert tuple(adapter.routes.shape) == (s, k)
        _assert_all(torch.equal(adapter.owner.cpu().long(), owners[rank]), "owner differs from encoded dst")
        support = adapter.support.cpu().bool()
        _assert_all(bool(support[owners, routes].all()), "execution uses unsupported expert")
        for dest in range(R):
            assert bool(support[dest, dest*epn:(dest+1)*epn].all())
            assert int(support[dest].sum())-epn <= b
        initial = plan_metrics(gather_raw(adapter.seed).reshape(R, s, k), routes, nvs, e)
        assert metrics["remote_token_destinations"] <= initial["remote_token_destinations"]
        # src_info is temporary provenance from this fresh plan. Saved-plan
        # reuse deliberately keeps the newer scratch and consumes owned dedup.
        expected_src = torch.full((nvs,), -1, dtype=torch.int32)
        src_values = (torch.arange(R)[:, None, None]*nvs
                      + torch.arange(s*k).view(1, s, k)).expand(R, s, k)
        local = owners == rank
        expected_src[offsets[local]] = src_values[local].int()
        begin = rank*int(c["meta_chunk_padded"])+int(c["SRC_INFO_OFF"])
        _assert_all(torch.equal(c["meta_buf"][begin:begin+nvs].cpu(), expected_src), "published source provenance mismatch")
    return routes, owners, offsets


def _check_payload(dispatched, hidden, owners, offsets, rank):
    all_hidden = _gather_cpu(hidden)
    local = owners == rank
    expected = all_hidden[:, :, None, :].expand(*owners.shape, H)[local]
    _assert_all(torch.equal(dispatched[offsets[local].to("cuda")].cpu(), expected), "dispatch source/token payload mismatch")


def _fresh(buffer, topk, tpe, cfg, version):
    c = buffer._require_ctx();rank, s, k = (int(c[x]) for x in ("rank", "S", "K"))
    x = _hidden(rank, s, version)
    result = buffer.dispatch(x, topk_experts_sk=topk, tokens_per_expert=tpe)
    torch.cuda.synchronize()
    rows, weights, cu, plan = result
    assert weights is None
    _, owners, offsets = _audit(buffer, topk, plan, cu, cfg)
    _check_payload(rows, x, owners, offsets, rank)
    out = buffer.combine(plan=plan, hidden_nvsh=rows)[0]
    torch.cuda.synchronize()
    _assert_all(torch.equal(out, x*k), "identity combine mismatch")
    return dict(plan=plan.clone(), cu=cu.clone(), topk=topk.clone())


def _reuse_after_new_plan(buffer, saved, cfg):
    c = buffer._require_ctx();rank, s, k = (int(c[x]) for x in ("rank", "S", "K"))
    plan = saved["plan"]
    snapshot = {name: getattr(plan, name).clone() for name in PLAN_FIELDS}
    x = _hidden(rank, s, 19)
    result = buffer.dispatch(x, plan=plan)
    torch.cuda.synchronize()
    rows, weights, cu, returned = result
    assert weights is None and cu is None and returned is plan
    _assert_all(all(torch.equal(getattr(plan, name), value) for name, value in snapshot.items()), "plan reuse overwrote saved metadata")
    _, owners, offsets = _audit(buffer, saved["topk"], plan, saved["cu"], cfg, current=False)
    _check_payload(rows, x, owners, offsets, rank)
    out = buffer.combine(plan=plan, hidden_nvsh=rows)[0]
    torch.cuda.synchronize()
    _assert_all(torch.equal(out, x*k), "saved-plan identity combine mismatch")


def _graph_changes(buffer, cfg, rank, label):
    e, k, s = (int(buffer._require_ctx()[key]) for key in ("E", "K", "S"))
    topk, tpe = _routes(rank, e, k, s, 0)
    x = _hidden(rank, s, 0)
    graph = None
    try:
        def run():
            rows, _, cu, plan = buffer.dispatch(x, topk_experts_sk=topk, tokens_per_expert=tpe)
            out = buffer.combine(plan=plan, hidden_nvsh=rows)[0]
            return plan, cu, out
        _log(rank, label, "warm graph path")
        for _ in range(3):
            run()
        _ready()
        _log(rank, label, "capture graph")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan, cu, out = run()
        for phase in (0, 1, 2):
            _log(rank, label, f"graph replay changed input phase={phase}")
            new_topk, new_tpe = _routes(rank, e, k, s, phase)
            topk.copy_(new_topk);tpe.copy_(new_tpe);x.copy_(_hidden(rank, s, phase))
            _ready()
            graph.replay()
            torch.cuda.synchronize()
            _audit(buffer, topk, plan, cu, cfg)
            _assert_all(torch.equal(out, x*k), "graph replay used stale inputs or wrong payload")
        _ready()
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()


def test_all_valid_but_different_rank_configuration_rejected_before_allocation(eight_physical, monkeypatch):
    rank = eight_physical
    def allocation_reached(*args, **kwargs):
        raise AssertionError("mismatched planner settings reached NVL allocation")
    monkeypatch.setattr(api, "_create_context", allocation_reached)
    monkeypatch.setattr(Buffer, "__del__", lambda self: None)
    cfg = SampledPlannerConfig(sample_size=37+rank, groups_per_class=1, rounds=1)
    _log(rank, "preflight", "reject independently valid, cross-rank different sample sizes")
    with pytest.raises(ValueError, match="must match on every EP rank"):
        Buffer(17, H, 1, 64, R, B=8, planning_backend="sampled_group", planner_config=cfg)
    _ready()
    _log(rank, "preflight", "PASS before allocation")


def test_one_invalid_rank_rejects_all_before_allocation(eight_physical, monkeypatch):
    rank = eight_physical
    monkeypatch.setattr(api, "_create_context", lambda *a, **k: pytest.fail("invalid settings reached allocation"))
    monkeypatch.setattr(Buffer, "__del__", lambda self: None)
    with pytest.raises(ValueError, match="invalid sampled_group configuration"):
        Buffer(17, H, 1, 64, R, B=8, planning_backend="sampled_group",
               planner_config={"sample_size": 0 if rank == 3 else 37})
    _ready()


def test_one_invalid_model_rank_rejects_all_before_allocation(eight_physical, monkeypatch):
    rank = eight_physical
    monkeypatch.setattr(api, "_create_context", lambda *a, **k: pytest.fail("invalid model reached allocation"))
    monkeypatch.setattr(Buffer, "__del__", lambda self: None)
    model = {"hidden_size": H, "num_experts": 64, "num_experts_per_tok": 1}
    if rank == 2:
        del model["num_experts_per_tok"]
    with pytest.raises(ValueError):
        Buffer.from_model_config(model, S=17, num_ep_ranks=R, planning_backend="sampled_group")
    _ready()


def test_one_invalid_allocation_setting_rejects_all(eight_physical, monkeypatch):
    rank = eight_physical
    monkeypatch.setattr(api, "_create_context", lambda *a, **k: pytest.fail("invalid setting reached allocation"))
    monkeypatch.setattr(Buffer, "__del__", lambda self: None)
    with pytest.raises(ValueError):
        Buffer(17, H, 1, 64, R, token_padding=128.0 if rank == 2 else 128,
               planning_backend="sampled_group")
    _ready()


@pytest.mark.parametrize("case", CASES, ids=[_name(case) for case in CASES])
def test_variable_model_and_planner_configuration(eight_physical, case):
    rank = eight_physical
    e, k, d, g, rounds, s, b = case
    cfg = SampledPlannerConfig(sample_size=d, groups_per_class=g, rounds=rounds, sample_seed=SEED)
    label = _name(case)
    _log(rank, label, "construct Buffer and compile sampled CUDA extensions")
    options = dict(B=b, token_padding=128, num_sms=32, enable_pdl=False,
                   planning_backend="sampled_group", planner_config=cfg, explicitly_destroy=True)
    if CASES.index(case) % 2 == 0:
        model = {"text_config": SimpleNamespace(d_model=H, n_routed_experts=e, moe_top_k=k)}
        buffer = Buffer.from_model_config(model, S=s, num_ep_ranks=R, **options)
    else:
        buffer = Buffer(s, H, k, e, R, **options)
    try:
        _compile_used(buffer, rank, label)
        saved = None
        for phase in (0, 1, 2):
            _log(rank, label, f"fresh local/hot runtime route phase={phase}")
            topk, tpe = _routes(rank, e, k, s, phase)
            result = _fresh(buffer, topk, tpe, cfg, phase)
            if phase == 0:
                saved = result
            if phase == 1:
                _log(rank, label, "reuse first saved plan after second fresh plan")
                _reuse_after_new_plan(buffer, saved, cfg)
        if (e, k, d) == (512, 16, 512):
            _graph_changes(buffer, cfg, rank, label)
        _ready()
        _log(rank, label, "PASS full layouts, payloads, dynamic routes, saved-plan lifetime")
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
