"""Candidate-limited selector and public eight-physical-GPU runtime checks.

Single-device structural oracle and generic/shared-cache coverage:
  python -m pytest -q -s tests/test_sampled_candidate_limit.py -k selector

Public runtime, changed CUDA Graph inputs, saved plans and collective fallback:
  torchrun --standalone --nproc_per_node=8 -m pytest -q -s \
    tests/test_sampled_candidate_limit.py -k 'not selector'

These use synthetic routes without private fixtures. They validate numerical
integration, not model-training performance or expert-gradient reduction.
"""
import os
import traceback

import numpy as np
import pytest
import torch
import torch.distributed as dist

import moonep.api as api
from moonep import Buffer, SampledPlannerConfig
from moonep.sampled_residency import Residency
from tests.sampled_residency_reference import reference
from tests.test_sampled_residency_configurable import structured_routes, assert_result
from tests.test_sampled_configurable import (
    eight_physical, _compile_used, _routes, _hidden, _fresh,
    _reuse_after_new_plan, _graph_changes, _gather_cpu, _assert_all, _ready,
)


SEED, R, H = 20261026, 8, 256


@pytest.mark.parametrize('experts,topk,samples,groups,table_path', [
    pytest.param(64, 1, 37, 1, True, id='limit128_clipped_to_capacity84'),
    pytest.param(352, 8, 256, 2, True, id='delivery_shared_cache'),
    pytest.param(128, 16, 128, 3, False, id='generic_shared_memory_fallback'),
])
def test_selector_candidate_limit_matches_oracle(experts, topk, samples, groups, table_path):
    if not torch.cuda.is_available():
        pytest.skip('one CUDA device is required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        pytest.skip('run selector tests with ordinary single-process pytest')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    s, slots, cap = 64, 8, 128
    planner = Residency(dict(R=R, E=experts, K=topk, S=s, B=slots, rank=0, device=device),
        SEED, sample_size=samples, groups_per_class=groups, candidate_limit=cap,
        component_only=True)
    assert planner.candidate_limit == cap
    assert planner.effective_candidate_limit == min(cap, 28*topk*3*groups**2)
    assert planner.uses_active_table == table_path
    ids = planner.ids.cpu().numpy()
    empty_copies = np.full((R, slots), -1, np.int32)
    copies = torch.as_tensor(empty_copies, device=device)
    routes = torch.empty((R, s, topk), device=device, dtype=torch.int32)
    owners = torch.empty_like(routes)
    route_views, owner_views = list(routes.unbind()), list(owners.unbind())
    packet_sum = torch.empty_like(planner.packet)

    def execute():
        packet_sum.zero_()
        for source in range(R):
            planner.gather_local(route_views[source], owner_views[source], copies, source_rank=source)
            packet_sum.add_(planner.packet)
        planner.build_from_reduced_samples(copies, packet_sum)

    def load(seed):
        x = structured_routes(experts, topk, ids, seed)
        initial = (x//(experts//R)).astype(np.int32)
        expected = reference(x, initial, empty_copies, ids, experts=experts,
            groups_per_class=groups, candidate_limit=cap)
        assert len(expected['packages']) > 0 and len(expected['history']) > 0
        if (experts,topk)==(352,8):
            assert len(expected['packages']) > cap, 'delivery fixture must actually prune'
        if planner.effective_candidate_limit < cap:
            unlimited = reference(x, initial, empty_copies, ids, experts=experts,
                groups_per_class=groups)
            np.testing.assert_array_equal(expected['support'], unlimited['support'])
            assert expected['history_rows'] == unlimited['history_rows']
        routes.copy_(torch.from_numpy(x).to(device))
        owners.copy_(torch.from_numpy(initial).to(device))
        return expected

    def check(expected, label):
        assert_result(planner, expected, label)
        diagnostics = planner.candidate_diagnostics()
        assert diagnostics['retained_P'] == min(cap, len(expected['packages']))
        np.testing.assert_array_equal(np.sort(diagnostics['selected_ids']), expected['retained_ids'])
        assert diagnostics['uses_active_table'] == table_path

    graph = None
    try:
        expected = load(932071)
        execute();torch.cuda.synchronize();check(expected, 'eager')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        expected = load(932072)
        planner.support.zero_();planner.frequencies.fill_(-1)
        planner.history_count.fill_(-1);planner.status.fill_(31)
        graph.replay();torch.cuda.synchronize();check(expected, 'graph changed input')
        # Invalid input must not consume stale active lists or select packages.
        copies[0,0] = experts
        graph.replay();torch.cuda.synchronize()
        assert int(planner.status.item()) != 0
        assert int(planner.history_count.item()) == 0
        copies.fill_(-1)
        graph.replay();torch.cuda.synchronize();check(expected, 'graph valid recovery')
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()


def test_different_candidate_limits_rejected_collectively(eight_physical, monkeypatch):
    rank = eight_physical
    monkeypatch.setattr(api, '_create_context', lambda *a, **k: pytest.fail('mismatch reached allocation'))
    monkeypatch.setattr(Buffer, '__del__', lambda self: None)
    cfg = SampledPlannerConfig(candidate_limit=128 if rank==3 else 0, rounds=4)
    with pytest.raises(ValueError, match='must match on every EP rank'):
        Buffer(17,H,1,64,R,B=8,planning_backend='sampled_group',planner_config=cfg)
    _ready()


def test_one_invalid_candidate_limit_rejected_collectively(eight_physical, monkeypatch):
    rank = eight_physical
    monkeypatch.setattr(api, '_create_context', lambda *a, **k: pytest.fail('invalid limit reached allocation'))
    monkeypatch.setattr(Buffer, '__del__', lambda self: None)
    with pytest.raises(ValueError, match='invalid sampled_group configuration'):
        Buffer(17,H,1,64,R,B=8,planning_backend='sampled_group',
            planner_config={'candidate_limit': -1 if rank==3 else 128})
    _ready()


def _oracle(buffer, topk, cfg, rank):
    adapter=buffer._require_ctx()['sampled_group_planner']
    routes=_gather_cpu(topk).numpy()
    initial=_gather_cpu(adapter.seed_owner).numpy()
    expected=[None]
    if rank==0:
        try:
            raw=reference(routes,initial,adapter.seed.experts_to_copy.cpu().numpy(),
                adapter.residency.ids.cpu().numpy(),experts=int(buffer._require_ctx()['E']),
                groups_per_class=cfg.groups_per_class,candidate_limit=cfg.candidate_limit)
            expected[0]=dict(passed=True,**{key:raw[key] for key in ('support','history','retained_ids')})
        except Exception:
            expected[0]=dict(passed=False,error=traceback.format_exc())
    dist.broadcast_object_list(expected,src=0)
    assert expected[0]['passed'],expected[0].get('error')
    _assert_all(np.array_equal(adapter.support.cpu().numpy(),expected[0]['support']),'CPU support mismatch')
    _assert_all(adapter.residency.accepted_packages()==expected[0]['history'],'CPU history mismatch')
    diagnostics=adapter.residency.candidate_diagnostics()
    _assert_all(np.array_equal(np.sort(diagnostics['selected_ids']),expected[0]['retained_ids']),'CPU retained candidates mismatch')


def _force_fallback(buffer, topk, tpe, cfg, rank):
    adapter=buffer._require_ctx()['sampled_group_planner']
    s,k=int(buffer._require_ctx()['S']),int(buffer._require_ctx()['K'])
    adapter.force_reject.fill_(int(rank==3))
    try:
        x=_hidden(rank,s,91)
        rows,_,cu,plan=buffer.dispatch(x,topk_experts_sk=topk,tokens_per_expert=tpe)
        torch.cuda.synchronize()
        _assert_all(int(adapter.status.item())!=0,'one rejecting rank must reject all')
        for field in ('dst','experts_to_copy','zero_fill_ranges','remote_stats'):
            _assert_all(torch.equal(getattr(plan,field),getattr(adapter.seed,field)),f'fallback changed {field}')
        _assert_all(torch.equal(cu,adapter.seed_cu),'fallback cumulative layout differs')
        out=buffer.combine(plan=plan,hidden_nvsh=rows)[0]
        torch.cuda.synchronize()
        _assert_all(torch.equal(out,x*k),'fallback payload/combine mismatch')
    finally:
        adapter.force_reject.zero_()
    _fresh(buffer,topk,tpe,cfg,92)
    _oracle(buffer,topk,cfg,rank)


@pytest.mark.parametrize('experts,topk,samples,groups,tokens,slots', [
    pytest.param(352,8,256,2,257,8,id='cap128_rounds4_delivery'),
    pytest.param(64,1,37,1,17,8,id='cap128_small_model_capacity84'),
])
def test_candidate_limit_public_runtime(eight_physical,experts,topk,samples,groups,tokens,slots):
    rank=eight_physical
    cfg=SampledPlannerConfig(sample_size=samples,groups_per_class=groups,
        candidate_limit=128,rounds=4,sample_seed=SEED)
    buffer=Buffer.from_model_config({'hidden_size':H,'num_experts':experts,'num_experts_per_tok':topk},
        S=tokens,num_ep_ranks=R,B=slots,enable_pdl=False,planning_backend='sampled_group',
        planner_config=cfg,explicitly_destroy=True)
    try:
        label=f'cap128_E{experts}_K{topk}'
        _compile_used(buffer,rank,label)
        saved=None
        for phase in (0,1,2):
            routes,tpe=_routes(rank,experts,topk,tokens,phase)
            item=_fresh(buffer,routes,tpe,cfg,phase)
            _oracle(buffer,routes,cfg,rank)
            if phase==0:saved=item
            elif phase==1:_reuse_after_new_plan(buffer,saved,cfg)
        _graph_changes(buffer,cfg,rank,label)
        routes,tpe=_routes(rank,experts,topk,tokens,1)
        _force_fallback(buffer,routes,tpe,cfg,rank)
        _ready()
    finally:
        torch.cuda.synchronize();buffer.destroy()


def test_candidate_limit_weighted_forward_backward(eight_physical,monkeypatch):
    """B8 training-style saved plans survive a later local-to-remote replan."""
    from tests.test_sampled_group import (
        _precompile_native, _routes as weighted_routes,
        _forward, _backward_after_new_plan,
    )
    rank=eight_physical
    cfg=SampledPlannerConfig(candidate_limit=128,rounds=4)
    buffer=Buffer(257,H,8,352,R,B=8,enable_pdl=False,
        planning_backend='sampled_group',planner_config=cfg,explicitly_destroy=True)
    try:
        _precompile_native(buffer)
        adapter=buffer._require_ctx()['sampled_group_planner']
        calls=[];run=adapter.run
        def counted(*args,**kwargs):
            calls.append(1)
            return run(*args,**kwargs)
        monkeypatch.setattr(adapter,'run',counted)
        first=_forward(buffer,*weighted_routes(rank,0,local_only=True))
        second=_forward(buffer,*weighted_routes(rank,1))
        _assert_all(len(calls)==2,'fresh forward must invoke capped planner exactly once')
        _assert_all(not torch.equal(first['topk'],second['topk']),'fixture needs two distinct microbatches')
        _backward_after_new_plan(buffer,first)
        _backward_after_new_plan(buffer,second)
        _assert_all(len(calls)==2,'saved-plan backward reran the planner')
        _ready()
    finally:
        torch.cuda.synchronize();buffer.destroy()
