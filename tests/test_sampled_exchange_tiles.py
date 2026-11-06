"""Exact eight-logical-source exchanges on one GPU; no private fixture needed.

python -m pytest -q -s tests/test_sampled_exchange_tiles.py

These tests compare canonical token ordinals and final owners, not private
histogram buffers whose shape intentionally changes with tile size.
"""
import os

import numpy as np
import pytest
import torch

from tests.benchmark_sampled_exchange_tiles import (
    R, ROUNDS, VARIANTS, synthetic_inputs, input_checks, make_team,
    execute_team, run_case, round_snapshot, assert_same, audit_round, host,
)


@pytest.mark.parametrize('tokens,topk', [
    pytest.param(1, 1, id='single_token_reciprocal'),
    pytest.param(31, 3, id='partial_warp_nonpower_k'),
    pytest.param(65, 8, id='partial_64_tile'),
    pytest.param(129, 16, id='partial_128_tile_generic_histogram'),
    pytest.param(257, 32, id='partial_256_tile_choice_bit31'),
])
def test_exchange_tiles_exact_reciprocal_graph(tokens, topk):
    if not torch.cuda.is_available():
        pytest.skip('one CUDA GPU is required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        pytest.skip('run using ordinary single-process pytest')
    torch.cuda.set_device(0)
    if torch.cuda.get_device_properties(0).multi_processor_count < 32:
        pytest.skip('matrix requires at least 32 SMs')
    routes_cpu, seed_cpu, support_cpu = synthetic_inputs(tokens, topk, phase=0)
    assert input_checks(routes_cpu, seed_cpu, support_cpu)['initially_equal']
    expected, baseline = run_case(routes_cpu, seed_cpu, support_cpu, tile=256, fuse=False, sm_limit=0)
    # A fixture producing no accepted exchanges would not exercise apply/quota.
    assert sum(row['audit']['stats'][0] for row in baseline['rounds']) > 0
    for tile, fuse in VARIANTS:
        run_case(routes_cpu, seed_cpu, support_cpu, tile=tile, fuse=fuse, sm_limit=32, expected=expected)

    # Reuse each captured graph with a new route, support, and seed in the same
    # storage. Poisoning exposes stale virtual columns and incomplete tail work.
    routes = torch.from_numpy(routes_cpu).cuda()
    support = torch.from_numpy(support_cpu).cuda()
    seed = torch.from_numpy(seed_cpu).cuda()
    teams, gathered, graphs = [], [], []
    try:
        for tile, fuse in VARIANTS:
            team = make_team(routes, support, tile=tile, fuse=fuse, sm_limit=32)
            counts = torch.empty((R, team[0].buckets), device='cuda', dtype=torch.int32)
            execute_team(team, seed, counts); torch.cuda.synchronize()
            assert_same(round_snapshot(team, ROUNDS - 1), expected[-1], 'before Graph capture')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute_team(team, seed, counts)
            teams.append(team); gathered.append(counts); graphs.append(graph)
        for phase in (1, 2):
            routes_cpu, seed_cpu, support_cpu = synthetic_inputs(tokens, topk, phase=phase)
            checks = input_checks(routes_cpu, seed_cpu, support_cpu)
            assert checks['initially_equal']
            expected, _ = run_case(routes_cpu, seed_cpu, support_cpu, tile=256, fuse=False, sm_limit=0)
            routes.copy_(torch.from_numpy(routes_cpu)); seed.copy_(torch.from_numpy(seed_cpu))
            support.copy_(torch.from_numpy(support_cpu))
            for team, counts, graph in zip(teams, gathered, graphs):
                counts.fill_(-17)
                for planner in team:
                    for field in ('owner', 'old_masks', 'keys', 'moves', 'ordinals', 'block_counts',
                                  'block_prefix', 'counts', 'total', 'source_prefix', 'accept', 'stats'):
                        getattr(planner, field).fill_(-9)
                graph.replay(); torch.cuda.synchronize()
                actual = round_snapshot(team, ROUNDS - 1)
                assert_same(actual, expected[-1], f'Graph changed phase {phase}')
                owner = np.stack([host(p.owner) for p in team])
                np.testing.assert_array_equal(np.bincount(owner.reshape(-1), minlength=R), checks['loads'])
                assert support_cpu[owner, routes_cpu].all()
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
