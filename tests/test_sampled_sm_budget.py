"""Synthetic correctness checks for the bounded and fused EP8 planner.

Single GPU exchange comparison:
  python -m pytest -q -s tests/test_sampled_sm_budget.py -k 'exchange or selector'

Eight distinct physical GPUs (no private router fixture needed):
  torchrun --standalone --nproc_per_node=8 -m pytest -q -s \
    tests/test_sampled_sm_budget.py -k 'not exchange and not selector'

These checks establish exact planner/communication behavior, not SM activity
measurement, training throughput, or expert-parameter gradient reduction.
"""
import os

import pytest
import torch

import moonep.api as api
from moonep import Buffer, SampledPlannerConfig
from moonep.sampled_exchange import SampledExchange
from moonep.sampled_runtime import BoundedPlannerRuntime
from tests.test_sampled_configurable import (
    eight_physical, _compile_used, _routes, _fresh, _reuse_after_new_plan,
    _graph_changes, _assert_all, _ready,
)
from tests.test_sampled_candidate_limit import _oracle, _force_fallback

R, H, SEED = 8, 256, 20261026


@pytest.mark.parametrize('tokens,topk', [
    pytest.param(1, 1, id='single_token'),
    pytest.param(31, 8, id='partial_subwarp_tile'),
    pytest.param(255, 16, id='partial_histogram_tile'),
    pytest.param(257, 32, id='choice_bit31_and_tail'),
    pytest.param(16768, 8, id='delivery_local_token_count'),
])
def test_exchange_bounded_fused_exact(tokens, topk):
    if not torch.cuda.is_available():
        pytest.skip('one CUDA device is required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        pytest.skip('run exchange checks with ordinary single-process pytest')
    torch.cuda.set_device(0)
    if torch.cuda.get_device_properties(0).multi_processor_count < 32:
        pytest.skip('this matrix requires a device with at least 32 SMs')
    experts = 352 if topk == 8 else max(64, R*topk)
    routes = torch.empty((tokens, topk), device='cuda', dtype=torch.int32)
    support = torch.empty((R, experts), device='cuda', dtype=torch.bool)
    seed = torch.empty_like(routes)
    baseline = SampledExchange(routes, support, source=3, rounds=4)
    variants = [SampledExchange(routes, support, source=3, rounds=4,
                sm_limit=limit, fuse=fuse)
                for limit, fuse in ((1, False), (1, True), (32, False), (32, True))]
    planners = [baseline]+variants
    gathered = [torch.empty((R, p.buckets), device='cuda', dtype=torch.int32) for p in planners]
    fields = ('resident', 'exec_masks', 'owner', 'old_masks', 'keys', 'moves',
              'ordinals', 'block_counts', 'block_prefix', 'counts', 'total',
              'source_prefix', 'accept', 'stats')
    graphs = []

    def load(phase):
        token = torch.arange(tokens, device='cuda', dtype=torch.int64)[:, None]
        choice = torch.arange(topk, device='cuda', dtype=torch.int64)[None, :]
        value = ((token*topk+choice+phase*13) % experts).int()
        routes.copy_(value)
        seed.copy_(value//(experts//R))
        e = torch.arange(experts, device='cuda')[None, :]
        rank = torch.arange(R, device='cuda')[:, None]
        support.copy_((rank == e//(experts//R)) | ((e+rank*3+phase) % 5 < 3))

    def execute(index):
        p = planners[index]
        p.prepare_support();p.reset(seed)
        for step in range(4):
            counts = p.propose(step)
            # Synthetic remote sources intentionally supply ample reciprocal
            # candidates. This isolates exact local quota/prefix behavior;
            # real matched source counts are covered by the eight-GPU tests.
            gathered[index].fill_(100000)
            gathered[index][p.source].copy_(counts)
            p.commit(step, gathered[index])

    def snapshot():
        return {name: getattr(baseline, name).clone() for name in fields}

    def compare(p, expected, label):
        for name in fields:
            torch.testing.assert_close(getattr(p, name), expected[name], rtol=0, atol=0,
                                       msg=f'{label}: {name}')

    try:
        load(0)
        execute(0)
        expected = snapshot()
        for i, p in enumerate(variants, 1):
            execute(i);torch.cuda.synchronize()
            compare(p, expected, f'eager limit={p.sm_limit} fuse={p.fuse}')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute(i)
            graphs.append(graph)
        for phase in (1, 2):
            load(phase)
            execute(0)
            expected = snapshot()
            for p, graph in zip(variants, graphs):
                # A truncated physical grid leaves stale virtual columns;
                # poisoning catches that even when a previous value was zero.
                p.block_counts.fill_(-9);p.block_prefix.fill_(-7)
                p.owner.fill_(-1);p.keys.fill_(-1);p.stats.fill_(-1)
                graph.replay();torch.cuda.synchronize()
                compare(p, expected, f'graph phase={phase} limit={p.sm_limit} fuse={p.fuse}')
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize('experts,topk,samples,groups,table_path,fuse', [
    pytest.param(64, 1, 37, 1, True, False, id='small_clipped_limit'),
    pytest.param(352, 8, 256, 2, True, False, id='delivery_unfused'),
    pytest.param(352, 8, 256, 2, True, True, id='delivery_fused'),
    pytest.param(128, 16, 128, 3, False, True, id='generic_fused'),
])
def test_selector_bounded_matches_cpu_oracle(monkeypatch, experts, topk, samples,
                                           groups, table_path, fuse):
    import tests.test_sampled_candidate_limit as candidate_tests
    from moonep.sampled_residency import Residency
    # Retain the independent CPU oracle, all intermediate table/history checks,
    # poisoned workspace, changed-input Graph, and malformed-copy recovery.
    def residency(*args, **kwargs):
        return Residency(*args, **kwargs, sm_limit=32, fuse=fuse)
    monkeypatch.setattr(candidate_tests, 'Residency', residency)
    candidate_tests.test_selector_candidate_limit_matches_oracle(
        experts, topk, samples, groups, table_path)


@pytest.mark.parametrize('words,fuse', [
    pytest.param(1, False, id='scalar_unfused'),
    pytest.param(4609, False, id='sample_packet_unfused'),
    pytest.param(1, True, id='scalar_fused'),
    pytest.param(4609, True, id='sample_packet_fused'),
    pytest.param(8193, True, id='multiple_tiles_and_tail_fused'),
])
def test_peer_collectives_repeated_graph(eight_physical, words, fuse):
    rank = eight_physical
    device = torch.device('cuda', torch.cuda.current_device())
    ctx = dict(rank=rank, group=None, meta_buf=torch.empty(1, device=device, dtype=torch.int32))
    runtime = BoundedPlannerRuntime(ctx, sm_limit=32, fuse=fuse, max_words=words)
    src = torch.empty(words, device=device, dtype=torch.int32)
    summed, maximum = torch.empty_like(src), torch.empty_like(src)
    gathered = torch.empty((R, words), device=device, dtype=torch.int32)
    graph = None

    def load(phase):
        src.copy_((torch.arange(words, device=device, dtype=torch.int32)*3
                   + rank*100000+phase*17)-400000)

    def execute():
        runtime.all_gather(src, gathered)
        runtime.copy(src, summed);runtime.all_reduce(summed, 'sum')
        runtime.copy(src, maximum);runtime.all_reduce(maximum, 'max')

    def check(phase):
        rows = (torch.arange(words, device=device, dtype=torch.int32)[None, :]*3
                + torch.arange(R, device=device, dtype=torch.int32)[:, None]*100000
                + phase*17)-400000
        _assert_all(torch.equal(gathered, rows), 'peer all_gather differs from ordered source rows')
        _assert_all(torch.equal(summed, rows.sum(0).int()), 'peer in-place SUM mismatch')
        _assert_all(torch.equal(maximum, rows.max(0).values), 'peer in-place MAX mismatch')

    try:
        # Do not let any rank enter the peer barrier while another is JITing.
        _ready()
        load(0);execute();torch.cuda.synchronize();check(0)
        _ready()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        # More than four invocations exercise the alternating epoch/sign wrap.
        for phase in range(1, 9):
            load(phase)
            gathered.fill_(123);summed.fill_(123);maximum.fill_(123)
            _ready();graph.replay();torch.cuda.synchronize();check(phase)
        _ready()
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        _ready()
        del runtime
        _ready()


@pytest.mark.parametrize('change', ['sm_limit', 'fuse'])
def test_sm_budget_mismatch_rejected_collectively(eight_physical, monkeypatch, change):
    rank = eight_physical
    monkeypatch.setattr(api, '_create_context', lambda *a, **k: pytest.fail('mismatch reached allocation'))
    monkeypatch.setattr(Buffer, '__del__', lambda self: None)
    values = dict(candidate_limit=128, rounds=4, sm_limit=32, fuse=False)
    if rank == 3:
        values[change] = 1 if change == 'sm_limit' else True
    with pytest.raises(ValueError, match='must match on every EP rank'):
        Buffer(17, H, 1, 64, R, B=8, planning_backend='sampled_group', planner_config=values)
    _ready()


def test_sm_budget_above_device_rejected_collectively(eight_physical, monkeypatch):
    rank = eight_physical
    monkeypatch.setattr(api, '_create_context', lambda *a, **k: pytest.fail('invalid SM limit reached allocation'))
    monkeypatch.setattr(Buffer, '__del__', lambda self: None)
    limit = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count+1 if rank == 3 else 32
    with pytest.raises(ValueError, match='invalid sampled_group configuration'):
        Buffer(17, H, 1, 64, R, B=8, planning_backend='sampled_group',
               planner_config=dict(candidate_limit=128, rounds=4, sm_limit=limit))
    _ready()


def _graph_rejection_and_recovery(buffer, cfg, rank):
    from moonep.planning import allocate_planning_outputs, launch_planning
    from moonep.inter_rank_sync import launch_inter_rank_sync
    ctx = buffer._require_ctx();adapter = ctx['sampled_group_planner']
    e, k, s = (int(ctx[name]) for name in ('E', 'K', 'S'))
    topk, tpe = _routes(rank, e, k, s, 1)
    plan, cu = allocate_planning_outputs(ctx)
    graph = None
    def execute():
        launch_inter_rank_sync(ctx)
        launch_planning(ctx, topk.flatten(), tpe, cu, plan)
    try:
        execute();_ready()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        adapter.force_reject.fill_(int(rank == 3))
        graph.replay();torch.cuda.synchronize()
        _assert_all(int(adapter.status.item()) != 0, 'graph failed to propagate one-rank rejection')
        for field in ('dst', 'experts_to_copy', 'zero_fill_ranges', 'remote_stats'):
            _assert_all(torch.equal(getattr(plan, field), getattr(adapter.seed, field)), f'graph fallback changed {field}')
        _assert_all(torch.equal(cu, adapter.seed_cu), 'graph fallback changed cumulative layout')
        adapter.force_reject.zero_()
        graph.replay();torch.cuda.synchronize()
        _assert_all(int(adapter.status.item()) == 0, 'graph valid replay did not recover')
        _oracle(buffer, topk, cfg, rank)
        _ready()
    finally:
        adapter.force_reject.zero_()
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize('experts,topk,samples,groups,tokens,slots,fuse', [
    pytest.param(352, 8, 256, 2, 257, 8, False, id='delivery32_unfused'),
    pytest.param(352, 8, 256, 2, 257, 8, True, id='delivery32_fused'),
    pytest.param(64, 1, 37, 1, 17, 8, True, id='small_dynamic_model'),
    pytest.param(128, 16, 128, 3, 33, 16, True, id='generic_residency_and_topk16'),
    pytest.param(64, 3, 37, 1, 17, 8, True, id='nonpower_topk3_partial_tiles'),
])
def test_sm_budget_public_runtime(eight_physical, experts, topk, samples, groups, tokens, slots, fuse):
    rank = eight_physical
    cfg = SampledPlannerConfig(sample_size=samples, groups_per_class=groups,
        candidate_limit=128, rounds=4, sample_seed=SEED, sm_limit=32, fuse=fuse)
    buffer = Buffer.from_model_config(
        {'hidden_size': H, 'num_experts': experts, 'num_experts_per_tok': topk},
        S=tokens, num_ep_ranks=R, B=slots, enable_pdl=False,
        planning_backend='sampled_group', planner_config=cfg, explicitly_destroy=True)
    try:
        label = f'SM32_E{experts}_K{topk}_fuse{fuse}'
        _compile_used(buffer, rank, label)
        saved = None
        for phase in (0, 1, 2):
            routes, tpe = _routes(rank, experts, topk, tokens, phase)
            item = _fresh(buffer, routes, tpe, cfg, phase)
            _oracle(buffer, routes, cfg, rank)
            if phase == 0:
                saved = item
            elif phase == 1:
                _reuse_after_new_plan(buffer, saved, cfg)
        _graph_changes(buffer, cfg, rank, label)
        routes, tpe = _routes(rank, experts, topk, tokens, 1)
        _force_fallback(buffer, routes, tpe, cfg, rank)
        _graph_rejection_and_recovery(buffer, cfg, rank)
        _ready()
    finally:
        torch.cuda.synchronize();buffer.destroy()


@pytest.mark.parametrize('fuse', [False, True], ids=['unfused', 'fused'])
def test_sm_budget_weighted_forward_backward(eight_physical, monkeypatch, fuse):
    import tests.test_sampled_candidate_limit as candidate_tests
    # Reuse the established two-microbatch forward/backward numerical oracle;
    # only the explicit planner configuration changes in this coverage lane.
    def configuration(**values):
        return SampledPlannerConfig(**values, sm_limit=32, fuse=fuse)
    monkeypatch.setattr(candidate_tests, 'SampledPlannerConfig', configuration)
    candidate_tests.test_candidate_limit_weighted_forward_backward(eight_physical, monkeypatch)
