"""One-GPU exchange-stage microbenchmark, with eight LOGICAL source ranks.

python -m tests.benchmark_sampled_exchange_tiles --fixture-root fixtures \
  --output evidence/exchange_tiles_01.json

All eight sources execute sequentially on one GPU for exact quota validation.
The fuse=True microbenchmark explicitly forces LEGACY proposal fusion; the
production selective policy leaves proposals unfused, even when quota is fused.
Only one source's named CUDA Graph stage is timed at a time; these timings are
NOT eight-physical-GPU planner latency. Frozen fixture support/seed are inputs
to this exchange component, not a live cap128 residency/planner measurement.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from moonep.sampled_exchange import SampledExchange

R, ROUNDS, SEED = 8, 4, 20261026
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [(tile, fuse) for tile in (256, 128, 64) for fuse in (False, True)]


def array_hash(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def host(value):
    return value.detach().cpu().numpy().copy()


def synthetic_inputs(tokens, topk, phase=0):
    """Legal home seed with exactly S*K pairs/rank, and paired replicas.

    Each source is a permutation of the eight owners for every (token,choice),
    hence arbitrary S and K retain exact global load balance. K32 includes
    choice bit 31 and repeated owner groups. Routes contain no duplicate expert.
    """
    experts = max(64, R * topk)
    per_rank = experts // R
    peer = 1 + 2 * (phase % 2)
    source = np.arange(R, dtype=np.int64)[:, None, None]
    token = np.arange(tokens, dtype=np.int64)[None, :, None]
    choice = np.arange(topk, dtype=np.int64)[None, None, :]
    seed = ((source ^ peer) + token + choice) % R
    # Repeated choices assigned to one owner use adjacent local experts. Some
    # complete K32 owner groups therefore fit in the eight replica slots, so
    # the high-choice-bit test exercises accepted moves rather than only zeros.
    routes = seed * per_rank + (token * 17 + choice // R + phase * 5) % per_rank
    expert = np.arange(experts)[None, :]
    rank = np.arange(R)[:, None]
    support = ((rank == expert // per_rank) |
               ((rank ^ peer) == expert // per_rank) & (expert % per_rank < min(8, per_rank)))
    return routes.astype(np.int32), seed.astype(np.int32), support.astype(np.bool_)


def input_checks(routes, seed, support):
    assert routes.ndim == 3 and routes.shape[0] == R and routes.shape == seed.shape
    assert routes.dtype == seed.dtype == np.int32 and support.dtype == np.bool_
    assert support.ndim == 2 and support.shape[0] == R
    assert 1 <= routes.shape[2] <= 32 and routes.shape[1] > 0
    assert routes.min() >= 0 and routes.max() < support.shape[1]
    assert seed.min() >= 0 and seed.max() < R
    assert (np.diff(np.sort(routes, axis=-1), axis=-1) != 0).all()
    assert support[seed, routes].all(), 'initial seed executes an absent expert'
    loads = np.bincount(seed.reshape(-1), minlength=R)
    return dict(loads=loads.tolist(), initially_equal=bool((loads == loads[0]).all()),
                routes=array_hash(routes), seed=array_hash(seed), support=array_hash(support))


def remote_destinations(owner):
    return sum(int((owner[source] == destination).any(-1).sum())
               for source in range(R) for destination in range(R) if source != destination)


def make_team(routes, support, *, tile=256, fuse=False, sm_limit=0):
    return [SampledExchange(routes[source], support, source=source, rounds=ROUNDS,
                            sm_limit=sm_limit, fuse=fuse, _tile_tokens=tile, _fuse_proposals=fuse)
            for source in range(R)]


def proposal_snapshot(planner):
    """Canonical ordinal ignores the intentionally variable histogram tiling."""
    keys = host(planner.keys)
    ordinals = host(planner.ordinals)
    prefix = host(planner.block_prefix)
    valid = keys < planner.buckets
    assert (keys >= 0).all()
    canonical = np.full(planner.S, -1, dtype=np.int32)
    token = np.arange(planner.S)[valid]
    canonical[valid] = ordinals[valid] + prefix[keys[valid], token // planner._tile_tokens]
    counts = host(planner.counts)
    # Independent stable-order oracle: scan tokens once in original order.
    seen = np.zeros(planner.buckets, dtype=np.int32)
    expected = np.full(planner.S, -1, dtype=np.int32)
    for t, key in enumerate(keys):
        if key < planner.buckets:
            expected[t] = seen[key]
            seen[key] += 1
    np.testing.assert_array_equal(canonical, expected, err_msg='unstable source-token order')
    np.testing.assert_array_equal(counts, seen, err_msg='proposal histogram differs from keys')
    return dict(keys=keys, moves=host(planner.moves), source_ordinal=canonical, counts=counts)


def round_snapshot(team, step):
    rows = []
    for planner in team:
        row = proposal_snapshot(planner)
        for key in ('owner', 'old_masks', 'exec_masks', 'resident', 'total', 'source_prefix', 'accept'):
            row[key] = host(getattr(planner, key))
        row['stats'] = host(planner.stats[:step + 1])
        rows.append(row)
    return rows


def assert_same(actual, expected, label):
    assert len(actual) == len(expected) == R
    for rank, (a, e) in enumerate(zip(actual, expected)):
        assert set(a) == set(e)
        for field in a:
            np.testing.assert_array_equal(a[field], e[field], err_msg=f'{label} source{rank}: {field}')


def audit_round(rows, previous, routes, support, initial_loads):
    owner = np.stack([row['owner'] for row in rows])
    assert owner.min() >= 0 and owner.max() < R
    assert support[owner, routes].all(), 'assigned expert not resident'
    np.testing.assert_array_equal(np.bincount(owner.reshape(-1), minlength=R), initial_loads)
    all_counts = np.stack([row['counts'] for row in rows])
    total = all_counts.sum(0).astype(np.int32)
    for source, row in enumerate(rows):
        np.testing.assert_array_equal(row['total'], total)
        np.testing.assert_array_equal(row['source_prefix'], all_counts[:source].sum(0))
        np.testing.assert_array_equal(row['accept'], rows[0]['accept'])
        np.testing.assert_array_equal(row['stats'], rows[0]['stats'])
        valid = row['keys'] < len(total)
        accepted = np.zeros(len(valid), dtype=bool)
        key = row['keys'][valid]
        accepted[valid] = (row['source_ordinal'][valid] + row['source_prefix'][key] < row['accept'][key])
        np.testing.assert_array_equal(accepted, (owner[source] != previous[source]).any(-1))
    stats = rows[0]['stats'][-1]
    changed_pairs = int((owner != previous).sum())
    changed_tokens = int((owner != previous).any(-1).sum())
    before, after = remote_destinations(previous), remote_destinations(owner)
    assert changed_pairs == int(stats[1]), 'moved-pair stats disagree with actual owners'
    assert changed_tokens == int(stats[2]) == 2 * int(stats[0])
    assert before - after == int(stats[3]) >= 0, 'predicted gain disagrees with actual remote destinations'
    return owner, dict(loads=np.bincount(owner.reshape(-1), minlength=R).tolist(),
                       before_remote=before, after_remote=after, stats=stats.tolist())


def execute_team(team, seed, gathered):
    """Capture-friendly sequential emulation of real reciprocal count exchange."""
    for rank, planner in enumerate(team):
        planner.prepare_support(); planner.reset(seed[rank])
    for step in range(ROUNDS):
        for planner in team:
            planner.propose(step)
        torch.stack([planner.counts for planner in team], out=gathered)
        for planner in team:
            planner.commit(step, gathered)


def measure_stage(planner, step, stage, samples, replays):
    fn = lambda: planner.propose(step, _stage=stage)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    rows = []
    try:
        with torch.cuda.graph(graph):
            fn()
        graph.replay(); torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for sample in range(samples):
            begin.record()
            for _ in range(replays):
                graph.replay()
            end.record(); end.synchronize()
            elapsed = begin.elapsed_time(end)
            rows.append(dict(sample=sample, replays=replays, event_total_ms=elapsed,
                             cuda_us=elapsed * 1000 / replays))
        assert all(math.isfinite(row['cuda_us']) and row['cuda_us'] > 0 for row in rows)
    finally:
        torch.cuda.synchronize(); graph.reset()
    return dict(stage=stage, raw=rows, median_cuda_us=statistics.median(x['cuda_us'] for x in rows))


def run_case(routes_cpu, seed_cpu, support_cpu, *, tile, fuse, sm_limit,
             sources=(), samples=10, replays=20, expected=None, saved_owner=None, saved_stats=None):
    checks = input_checks(routes_cpu, seed_cpu, support_cpu)
    routes = torch.from_numpy(routes_cpu).cuda(); support = torch.from_numpy(support_cpu).cuda()
    seed = torch.from_numpy(seed_cpu).cuda()
    team = make_team(routes, support, tile=tile, fuse=fuse, sm_limit=sm_limit)
    gathered = torch.empty((R, team[0].buckets), device='cuda', dtype=torch.int32)
    for rank, planner in enumerate(team):
        planner.prepare_support(); planner.reset(seed[rank])
    resources = team[0].ext.kernel_resources(tile)
    previous = seed_cpu.copy()
    snapshots, rounds = [], []
    for step in range(ROUNDS):
        for planner in team:
            planner.propose(step)
        pre_measurement = [proposal_snapshot(p) for p in team]
        timings = {}
        for rank in sources:
            planner = team[rank]
            stages = [('proposal', 1), ('ordinal', 2), ('prefix', 3), ('full_propose', 0)]
            if fuse:
                stages.append(('fused_proposal_ordinal', 4))
            # Rotate stage order across rounds, preserve and report all samples.
            shift = step % len(stages)
            stages = stages[shift:] + stages[:shift]
            timings[str(rank)] = {name: measure_stage(planner, step, stage, samples, replays)
                                  for name, stage in stages}
            planner.propose(step)  # finalize current counts after any isolated stage
        assert_same([proposal_snapshot(p) for p in team], pre_measurement, 'timing mutated proposal outputs')
        torch.stack([planner.counts for planner in team], out=gathered)
        for planner in team:
            planner.commit(step, gathered)
        torch.cuda.synchronize()
        rows = round_snapshot(team, step)
        if expected is not None:
            assert_same(rows, expected[step], f'round{step} tile{tile} fused{fuse}')
        owner, audit = audit_round(rows, previous, routes_cpu, support_cpu, np.asarray(checks['loads']))
        if saved_owner is not None:
            np.testing.assert_array_equal(owner, saved_owner[step + 1], err_msg='saved CPU owner oracle')
        if saved_stats is not None:
            np.testing.assert_array_equal(rows[0]['stats'][-1], saved_stats[step], err_msg='saved CPU stats oracle')
        rounds.append(dict(round=step, timings=timings, audit=audit,
                           outputs_by_source=[{k: array_hash(v) for k, v in row.items()} for row in rows]))
        snapshots.append(rows); previous = owner
    proposal_tiles = (routes_cpu.shape[1] + 31) // 32
    max_proposal_parts = (math.ceil(team[0].blocks / min(sm_limit or team[0].blocks, team[0].blocks)) * (tile // 32)
                          if fuse else math.ceil(proposal_tiles / min(sm_limit or proposal_tiles, proposal_tiles)))
    result = dict(tile_tokens=tile, sm_limit=sm_limit, fuse=fuse, proposal_fuse=fuse,
                  input=checks, resources=resources,
                  rounds=rounds, exact_vs_reference=expected is not None,
                  saved_cpu_oracle_checked=saved_owner is not None,
                  virtual_histogram_tiles=team[0].blocks,
                  max_cta_proposal_subtiles=max_proposal_parts,
                  final_owner=array_hash(previous))
    return snapshots, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root', type=Path)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--tokens', default='16384,16768')
    parser.add_argument('--sources', default='0', help='logical sources timed independently; e.g. 0,1,2,3,4,5,6,7')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--replays', type=int, default=20)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'never overwrite a previous attempt'
    assert int(os.environ.get('WORLD_SIZE', '1')) == 1, 'ordinary single-process invocation required'
    assert args.samples > 0 and args.replays > 0
    sources = list(map(int, args.sources.split(',')))
    tokens = list(map(int, args.tokens.split(',')))
    assert len(set(sources)) == len(sources) and all(0 <= x < R for x in sources)
    assert len(set(tokens)) == len(tokens) and all(x > 0 for x in tokens)
    torch.cuda.set_device(args.device); torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(args.device)
    assert props.multi_processor_count >= 32
    record = dict(status='running', failures=[], cases=[], metadata=dict(
        physical_gpus=1, logical_sources=8, device=args.device, gpu=props.name,
        gpu_uuid=str(props.uuid), device_sms=props.multi_processor_count,
        torch=torch.__version__, cuda=torch.version.cuda, sm_limit=32,
        sampled_sources=sources, samples=args.samples, replays=args.replays,
        rounds=ROUNDS, topk=8, seed=SEED,
        source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), ROOT/'moonep/sampled_exchange.py', ROOT/'moonep/csrc/sampled_exchange.cu')},
        timing='One source stage at a time, CUDA Graph, CUDA events; no other logical source runs during the timed interval.',
        limitations=['Single GPU component timing, not eight-physical-GPU planner latency.',
                     'Frozen support/seed bypass residency; this experiment does not change or validate cap128 candidate search.',
                     'Counts are gathered sequentially with torch.stack for correctness, outside all measured stages.',
                     'Shortened real fixtures may start unbalanced; exact initial per-rank load is conserved.']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        full_routes = full_seed = support = saved_owner = saved_stats = None
        if args.fixture_root is not None:
            path = args.fixture_root / f'layer_{args.layer:02d}.npz'
            with np.load(path, allow_pickle=False) as z:
                full_routes=z['topk']; full_seed=z['seed']; support=z['support']
                saved_owner=z['owner_prefixes']; saved_stats=z['stats']
            assert full_routes.shape[0] == R and full_routes.shape[2] == 8
            record['metadata']['fixture'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                support_kind='archived full-candidate support; held fixed for exchange microbenchmarks')
        for size in tokens:
            if full_routes is None:
                routes, seed, fixed_support = synthetic_inputs(size, 8)
                oracle_owner = oracle_stats = None
            else:
                assert size <= full_routes.shape[1]
                routes = np.ascontiguousarray(full_routes[:, :size])
                seed = np.ascontiguousarray(full_seed[:, :size]); fixed_support = support
                full = size == full_routes.shape[1]
                oracle_owner = saved_owner if full else None
                oracle_stats = saved_stats if full else None
                if full:
                    assert input_checks(routes, seed, fixed_support)['initially_equal']
                    np.testing.assert_array_equal(saved_owner[0], seed)
            expected, baseline = run_case(routes, seed, fixed_support, tile=256, fuse=False, sm_limit=0,
                                         saved_owner=oracle_owner, saved_stats=oracle_stats)
            case = dict(tokens_per_source=size, total_tokens=R*size, reference=baseline, variants=[])
            record['cases'].append(case); save()
            for tile, fuse in VARIANTS:
                _, result = run_case(routes, seed, fixed_support, tile=tile, fuse=fuse, sm_limit=32,
                                     sources=sources, samples=args.samples, replays=args.replays, expected=expected,
                                     saved_owner=oracle_owner, saved_stats=oracle_stats)
                case['variants'].append(result); save()
                print(json.dumps(dict(tokens=size, tile=tile, fuse=fuse, passed=True,
                    source0_full_us=[r['timings'].get('0', {}).get('full_propose', {}).get('median_cuda_us')
                                     for r in result['rounds']])), flush=True)
        record['status']='complete'; save()
    except Exception:
        record['status']='failed'; record['failures'].append(traceback.format_exc()); save()
        raise


if __name__ == '__main__':
    main()
