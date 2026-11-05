"""Public sampled_group integration checks; no private router dump is required.

Configuration-only (no process group or CUDA allocation):
    python -m pytest -q tests/test_sampled_group.py -k configuration

ACTUAL eight-GPU, single-node NVLink tests (not run by the two-GPU experiment):
    torchrun --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_group.py

The numerical test explicitly executes forward dispatch/combine and backward
saved-plan dispatch/combine for a scalar linear expert. It checks dX and router
weight gradients, not expert parameter gradient reduction, GEMM or optimizer.
The nondefault group has the same eight members; a nonzero global group root
would require additional physical GPUs and is not claimed by this test.
"""

import inspect
import os
import socket

import pytest
import torch
import torch.distributed as dist

import moonep.api as api
from moonep import Buffer
from tests.affinity_utils import check_expert_rows, gather_raw, plan_metrics


R, E, K, H, S = 8, 352, 8, 256, 257
SEED = 20261026


def test_configuration_registration_and_shape_guards(monkeypatch):
    """Both acceptance and rejection happen before any GPU/NVL allocation."""
    parameters = inspect.signature(Buffer).parameters
    assert 'planning_backend' in parameters
    assert 'planning_rounds' in parameters
    assert 'planning_sample_seed' in parameters

    class AllocationReached(Exception):
        pass

    def allocation(*args, **kwargs):
        raise AllocationReached

    monkeypatch.setattr(api, '_create_context', allocation)
    # Failed constructor objects must not try to destroy an unallocated context.
    monkeypatch.setattr(Buffer, '__del__', lambda self: None)
    defaults = dict(S=S, H=H, K=K, E=E, num_ep_ranks=R, B=8,
                    planning_backend='sampled_group', planning_rounds=4,
                    planning_sample_seed=SEED, explicitly_destroy=True)
    for slots in (8, 44):
        with pytest.raises(AllocationReached):
            Buffer(**(defaults | {'B': slots}))
    for change in ({'num_ep_ranks': 4}, {'E': 256}, {'K': 4}, {'B': 0},
                   {'planning_rounds': -1}, {'planning_backend': 'unknown_backend'}):
        with pytest.raises(ValueError):
            Buffer(**(defaults | change))
    # Native remains the default; this catches accidental opt-in by import.
    assert parameters['planning_backend'].default == 'moon'


@pytest.fixture
def real_eight_ranks(dist_env):
    rank, world = dist_env
    if world != R:
        pytest.skip('sampled_group public runtime requires eight actual ranks; two GPUs cannot validate this test')
    assert int(os.environ.get('LOCAL_WORLD_SIZE', 0)) == R, 'single-node eight-GPU test only'
    assert torch.cuda.device_count() >= R, 'eight visible physical GPUs are required'
    identities = [None] * R
    dist.all_gather_object(identities, (socket.gethostname(), torch.cuda.current_device()))
    assert len(set(identities)) == R, 'multiple ranks must not share one physical GPU'
    return rank


def _buffer(slots=8, group=None):
    buffer = Buffer(S, H, K, E, R, B=slots, group=group, enable_pdl=False,
                    planning_backend='sampled_group', planning_rounds=4,
                    planning_sample_seed=SEED, explicitly_destroy=True)
    _precompile_native(buffer)
    return buffer


def _precompile_native(buffer):
    """No collective CUDA kernel launches until every rank finishes CuTe JIT.

    The original planner has a bounded device barrier. Letting one rank enter
    that barrier while another compiles a new dtype/reuse variant can time out.
    This uses the actual parent API signature (no replica_max argument).
    """
    from moonep import (planning, inter_rank_sync, dispatch, combine,
                        dispatch_epilogue, combine_prologue)
    c = buffer._require_ctx()
    R_, E_, B_, S_, K_, H_, NvS, NvSp, stride, sms, device = (
        int(c[k]) for k in ('R', 'E', 'B', 'S', 'K', 'H', 'NvS',
                           'NvS_padded', 'meta_chunk_padded', 'num_sms', 'device'))
    planning._get_compiled(R_, E_, B_, S_, K_, int(c['NvS_capacity']), NvS,
        int(c['num_vblocks']), stride, int(c['TPE_OFF']), int(c['PLAN_OFF']),
        int(c['BARRIER_OFF']), int(c['TOPK0_OFF']), int(c['ORDER_OFF']),
        int(c['ORDER0_OFF']), int(c['token_padding']), sms)
    inter_rank_sync._get_compiled(R_, stride, inter_rank_sync._num_threads_for_ranks(R_), device)
    for weights in (False, True):
        for fresh in (False, True):
            dispatch._get_compiled(H_, R_, S_, K_, E_+B_, NvS, NvSp, stride,
                int(c['SRC_INFO_OFF']), sms, weights, fresh, device, False)
        combine._get_compiled(H_, R_, S_, K_, NvS, NvSp, stride, sms, weights, device, False)
    dispatch_epilogue._get_compiled(H_, NvS, int(c['num_sms_dedup']), device, False)
    combine_prologue._get_compiled(H_, R_, NvS, int(c['num_sms_dedup']), device, False)
    torch.cuda.synchronize()
    dist.barrier(group=c.get('group'))


def _routes(rank, phase, *, local_only=False):
    token = torch.arange(S, device='cuda', dtype=torch.int64)[:, None]
    slot = torch.arange(K, device='cuda', dtype=torch.int64)[None, :]
    if local_only:
        # Parent native planner is not generally legal for B<E/R. An all-local
        # balanced seed makes the B=8 backend check independent of that limit.
        x = rank*44+((token*K+slot+phase*11) % 44)
    elif phase % 2:
        # All ranks target the same eight experts: many real replicas, with
        # exactly eight distinct expert IDs per token, feasible at B=8.
        x = (44*(phase % R)+slot).expand(S, K)
    else:
        x = ((rank*S+token)*K+slot+phase*44) % E
    x = x.to(torch.int32).contiguous()
    return x, torch.bincount(x.flatten().long(), minlength=E).int()


def _hidden(rank, offset=0):
    # Base-eight digits identify every source/token. Values in [0,7]/8 keep
    # the selected linear expert arithmetic exactly representable in BF16.
    token = torch.arange(S, device='cuda', dtype=torch.int64)+rank*S+offset
    columns = torch.arange(H, device='cuda', dtype=torch.int64) % 4
    return (((token[:, None] >> (columns[None, :]*3)) & 7).float()/8).to(torch.bfloat16).contiguous()


def _router_weights(rank):
    ids = torch.arange(S*K, device='cuda', dtype=torch.int64).view(S, K)+rank*S*K
    return ((ids % 2+1).float()/8).contiguous()


def _gather(x, group=None):
    pieces = [torch.empty_like(x) for _ in range(R)]
    dist.all_gather(pieces, x, group=group)
    return torch.stack(pieces)


def _wait(result):
    if result[-1] is not None and isinstance(result[-1], torch.cuda.Event):
        torch.cuda.current_stream().wait_event(result[-1])
    torch.cuda.synchronize()


def _audit(buffer, topk, plan, cu, expect_success=True):
    ctx = buffer._require_ctx()
    group, rank = ctx.get('group'), int(ctx['rank'])
    adapter = ctx['sampled_group_planner']
    assert adapter.rounds == 4 and adapter.sample_seed == SEED
    assert tuple(adapter.routes.shape) == (S, K), 'backend must retain source-local routes'
    assert torch.equal(adapter.routes, topk), 'stale routes from an earlier planner call'
    routes = _gather(topk, group).cpu().long()
    raw = gather_raw(plan, group).reshape(R, S, K)
    metrics = plan_metrics(raw, routes, int(ctx['NvS']), E)
    assert metrics['loads'] == [S*K]*R
    assert max(metrics['active_replicas_per_rank']) <= int(ctx['B'])
    check_expert_rows(plan, cu, routes, ctx)
    address = torch.where(raw < 0, -raw-1, raw).long()
    owners = address // int(ctx['NvS'])
    if expect_success:
        assert int(adapter.status.item()) == 0, 'unexpected native fallback in supported test input'
        assert torch.equal(adapter.owner.reshape(S, K).cpu().long(), owners[rank])
        support = adapter.support.cpu().bool()
        assert tuple(support.shape) == (R, E)
        assert bool(support[owners, routes].all()), 'execution uses an absent expert'
        home = torch.arange(E)//(E//R)
        for r in range(R):
            assert bool(support[r, home == r].all())
            assert int(support[r, home != r].sum()) <= int(ctx['B'])
        seed_metrics = plan_metrics(gather_raw(adapter.seed, group).reshape(R, S, K),
                                    routes, int(ctx['NvS']), E)
        assert metrics['remote_token_destinations'] <= seed_metrics['remote_token_destinations']
    pair_ids = torch.nonzero(owners.flatten() == rank).flatten()
    offsets = (address.flatten()[pair_ids] % int(ctx['NvS'])).to('cuda')
    return routes, pair_ids, offsets


def _forward(buffer, topk, tpe, *, asynchronous=False, expect_success=True):
    ctx = buffer._require_ctx(); rank = int(ctx['rank']); group = ctx.get('group')
    x, weights = _hidden(rank), _router_weights(rank)
    result = buffer.dispatch(x, weights, topk, tpe, async_finish=asynchronous)
    _wait(result)
    dispatched, routed_weights, cu, plan = result[:4]
    routes, pair_ids, offsets = _audit(buffer, topk, plan, cu, expect_success)
    all_x = _gather(x, group).reshape(R*S, H)
    all_weights = _gather(weights, group).flatten()
    ids = pair_ids.to('cuda')
    assert torch.equal(dispatched[offsets], all_x[ids//K])
    assert torch.equal(routed_weights[offsets], all_weights[ids])
    assert dispatched.data_ptr() != ctx['hidden_buf_local'].data_ptr()
    assert routed_weights.data_ptr() != ctx['weights_buf_local'].data_ptr()
    factor = (routes.flatten()[pair_ids] % 2+1).to('cuda').float()
    # y_e = router_weight_e * a_e * x, where a_e is 1 or 2.
    output = torch.zeros_like(dispatched)
    output[offsets] = (dispatched[offsets].float()*factor[:, None]*routed_weights[offsets, None]).to(torch.bfloat16)
    combined = buffer.combine(plan=plan, hidden_nvsh=output, async_finish=asynchronous)
    _wait(combined)
    coeff = ((topk % 2+1).float()*weights).sum(1, keepdim=True)
    torch.testing.assert_close(combined[0], (x.float()*coeff).to(torch.bfloat16), rtol=0, atol=0)
    return dict(plan=plan.clone(), cu=cu.clone(), x=x, topk=topk.clone(),
                dispatched=dispatched.clone(), router=routed_weights.clone(),
                weights=weights, factor=factor, offsets=offsets, pair_ids=ids,
                group=group, coeff=coeff)


def _backward_after_new_plan(buffer, saved, *, asynchronous=False):
    ctx = buffer._require_ctx(); rank = int(ctx['rank'])
    plan = saved['plan']
    fields = {name: getattr(plan, name).clone() for name in
              ('dst', 'experts_to_copy', 'zero_fill_ranges', 'remote_stats',
               'dup_groups', 'dup_loffs', 'dup_counts')}
    dy = _hidden(rank, offset=17)
    result = buffer.dispatch(dy, plan=plan, async_finish=asynchronous)
    _wait(result)
    scattered_dy, no_weights, no_cu, returned_plan = result[:4]
    assert no_weights is None and no_cu is None and returned_plan is plan
    for name, value in fields.items():
        assert torch.equal(getattr(plan, name), value), f'reuse mutated saved {name}'
    offsets, factor = saved['offsets'], saved['factor']
    expected_dy = _gather(dy, saved['group']).reshape(R*S, H)[saved['pair_ids']//K]
    assert torch.equal(scattered_dy[offsets], expected_dy)
    dx_rows = torch.zeros_like(scattered_dy)
    dx_rows[offsets] = (scattered_dy[offsets].float()*factor[:, None]*saved['router'][offsets, None]).to(torch.bfloat16)
    dw_rows = torch.zeros(scattered_dy.shape[0], device='cuda', dtype=torch.float32)
    dw_rows[offsets] = (saved['dispatched'][offsets].float()*scattered_dy[offsets].float()).sum(1)*factor
    combined = buffer.combine(plan=plan, hidden_nvsh=dx_rows,
                              route_weights_nvs=dw_rows, async_finish=asynchronous)
    _wait(combined)
    expected_dx = (dy.float()*saved['coeff']).to(torch.bfloat16)
    expected_dw = (saved['x'].float()*dy.float()).sum(1, keepdim=True)*(saved['topk'] % 2+1).float()
    torch.testing.assert_close(combined[0], expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(combined[1], expected_dw, rtol=0, atol=0)


@pytest.mark.parametrize('slots,asynchronous', [(8, False), (44, True)])
def test_dynamic_public_forward_backward(real_eight_ranks, monkeypatch, slots, asynchronous):
    rank = real_eight_ranks
    buffer = _buffer(slots)
    try:
        adapter = buffer._require_ctx()['sampled_group_planner']
        run = adapter.run
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return run(*args, **kwargs)

        monkeypatch.setattr(adapter, 'run', counted)
        first = _forward(buffer, *_routes(rank, 0, local_only=slots < 44), asynchronous=asynchronous)
        second = _forward(buffer, *_routes(rank, 1, local_only=slots < 44), asynchronous=asynchronous)
        assert len(calls) == 2, 'public dispatch did not select the new planner exactly once'
        assert not torch.equal(first['topk'], second['topk'])
        # Backward of microbatch A after planner/dispatch B must retain A's
        # activation, router weights, dst and plan-owned dedup map.
        _backward_after_new_plan(buffer, first, asynchronous=asynchronous)
        _backward_after_new_plan(buffer, second, asynchronous=asynchronous)
        assert len(calls) == 2, 'saved-plan backward incorrectly reran the planner'
    finally:
        torch.cuda.synchronize()
        buffer.destroy()


def test_nondefault_group_and_collective_fallback(real_eight_ranks):
    rank = real_eight_ranks
    group = dist.new_group(ranks=list(range(R)))
    buffer = _buffer(44, group)
    try:
        ctx = buffer._require_ctx(); adapter = ctx['sampled_group_planner']
        assert ctx['group'] is group
        local_rank = dist.get_rank(group)
        assert local_rank == rank
        _forward(buffer, *_routes(local_rank, 0))
        # Only one rank rejects; all ranks must take one consistent fallback.
        adapter.force_reject.fill_(int(local_rank == 3))
        rejected = _forward(buffer, *_routes(local_rank, 1), expect_success=False)
        assert int(adapter.status.item()) != 0
        for name in ('dst', 'experts_to_copy', 'zero_fill_ranges', 'remote_stats'):
            assert torch.equal(getattr(rejected['plan'], name), getattr(adapter.seed, name)), name
        assert torch.equal(rejected['cu'], adapter.seed_cu)
        _backward_after_new_plan(buffer, rejected)
        adapter.force_reject.zero_()
        _forward(buffer, *_routes(local_rank, 2))
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
        dist.destroy_process_group(group)


def test_graph_replay_reads_new_routes(real_eight_ranks):
    rank = real_eight_ranks
    buffer = _buffer(44)
    graph = None
    try:
        topk, tpe = _routes(rank, 0)
        x = _hidden(rank)

        def launch():
            result = buffer.dispatch(x, topk_experts_sk=topk, tokens_per_expert=tpe)
            out = buffer.combine(plan=result[3], hidden_nvsh=result[0])[0]
            return result[3], result[2], out

        for _ in range(3):
            launch()
        torch.cuda.synchronize(); dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan, cu, out = launch()
        for phase in (0, 1, 2):
            new_routes, new_counts = _routes(rank, phase)
            topk.copy_(new_routes); tpe.copy_(new_counts)
            graph.replay()
            torch.cuda.synchronize()
            _audit(buffer, topk, plan, cu)
            torch.testing.assert_close(out, x*K, rtol=0, atol=0)
            dist.barrier()
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        buffer.destroy()
