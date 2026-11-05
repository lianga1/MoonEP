"""Measure only prepared-seed, fixed-support bucket exchange; CUDA timing + wall.

single: one physical GPU holds all eight source ranks.
local: eight physical GPUs use independent source-local buckets.
coordinated: eight physical GPUs exchange 1152 int32 bucket counts each round.
The latter's owner output must exactly match the single/global CPU reference.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, platform, statistics, time
import numpy as np
import torch
import torch.distributed as dist
from bucket_cuda import BucketCudaPlanner, BUCKETS, extension


def remote_count(owner):
    r = owner.shape[0]
    visited = (owner[..., None] == np.arange(r)).any(axis=2)
    for rank in range(r):
        visited[rank, :, rank] = False
    return int(visited.sum())


def audit(topk, seed, owner, support, expected):
    assert np.array_equal(owner, expected), f'CPU mismatch pairs: {np.count_nonzero(owner != expected)}'
    assert support[owner, topk].all(), 'pair assigned without resident expert'
    assert np.array_equal(np.bincount(owner.ravel(), minlength=8), np.bincount(seed.ravel(), minlength=8))
    home = np.arange(support.shape[1]) // (support.shape[1] // 8)
    extras = (support & (np.arange(8)[:, None] != home[None, :])).sum(1)
    assert extras.max() <= 8
    before, after = remote_count(seed), remote_count(owner)
    assert after <= before
    return dict(cpu_owner_exact=True, residency_valid=True, loads_preserved=True,
                remote_before=before, remote_after=after, saved=before-after,
                saved_percent=100 * (before-after) / before,
                loads=np.bincount(owner.ravel(), minlength=8).tolist(), extras=extras.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['single', 'local', 'coordinated'], required=True)
    p.add_argument('--inputs', type=Path, default=Path(__file__).parent / 'inputs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layers', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--methods', nargs='+', default=['seed', 'visited', 'reciprocal'])
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--reps', type=int, default=25)
    p.add_argument('--tail-only', action='store_true')
    p.add_argument('--graph', action='store_true', help='capture the full seed copy and four rounds, including collectives')
    a = p.parse_args()
    distributed = a.mode != 'single'
    rank = int(os.environ.get('RANK', '0')) if distributed else 0
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if distributed else 0
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group('nccl')
        assert dist.get_world_size() == 8
    # Compile before timing; rank zero builds once and the shared cache is reused.
    if distributed:
        if rank == 0:
            extension()
        dist.barrier()
    extension()
    cases = ['tail'] if a.tail_only else ['tail'] + [f'layer{l:02d}_{m}' for l in a.layers for m in a.methods]
    rows = []
    metadata = dict(mode=a.mode, physical_gpus=8 if distributed else 1,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                    python=platform.python_version(), utc=datetime.now(timezone.utc).isoformat(),
                    scope='fixed-residency bucket exchange only; seed copy included; seed/support construction, route transfer, dispatch/combine and weight traffic excluded',
                    collective_per_round='all_gather 1152 int32 = 4608 input bytes per source' if a.mode == 'coordinated' else 'none',
                    rounds=4, eager=not a.graph, cuda_graph=a.graph,
                    event_scope=('captured GPU replay timeline including metadata collectives' if a.graph else
                                 'eager GPU timeline including host launch supply gaps; not pure kernel sum'),
                    warmup=a.warmup, repetitions=a.reps)
    for case in cases:
        path = a.inputs / f'{case}.npz'
        data = np.load(path, allow_pickle=False)
        topk, seed_np, support_np = data['topk'], data['seed'], data['support']
        expected = data['expected_local' if a.mode == 'local' else 'expected_global']
        s = topk.shape[1]
        x = np.ascontiguousarray(topk[rank] if distributed else topk.reshape(-1, 8))
        seed_slice = np.ascontiguousarray(seed_np[rank] if distributed else seed_np.reshape(-1, 8))
        routes = torch.tensor(x, dtype=torch.int32, device='cuda')
        seed = torch.tensor(seed_slice, dtype=torch.int32, device='cuda')
        support = torch.tensor(support_np, dtype=torch.bool, device='cuda')
        planner = BucketCudaPlanner(routes, support, per_source=s, rounds=4,
                                    global_token_offset=rank * s if distributed else 0,
                                    coordinated=a.mode == 'coordinated')
        graph = None
        if a.graph:
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            # NCCL and allocation paths are warmed on the stream used for capture.
            with torch.cuda.stream(capture_stream):
                for _ in range(max(3, a.warmup)):
                    planner.run(seed)
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            if distributed:
                dist.barrier()
                torch.cuda.synchronize()
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    planner.run(seed)
                runner = graph.replay
            except Exception as error:
                a.output.parent.mkdir(parents=True, exist_ok=True)
                failure = dict(metadata=metadata, case=case, rank=rank,
                               status='graph_capture_failed', error=repr(error))
                a.output.with_suffix(f'.rank{rank}.graph_failure.json').write_text(
                    json.dumps(failure, indent=2) + '\n')
                raise
        else:
            runner = lambda: planner.run(seed)
        runner()
        torch.cuda.synchronize()
        # Full owner collection happens only for correctness, outside every timer.
        if distributed:
            all_owner = torch.empty((8 * s, 8), dtype=torch.int32, device='cuda')
            dist.all_gather_into_tensor(all_owner, planner.owner)
            actual = all_owner.cpu().numpy().reshape(topk.shape)
        else:
            actual = planner.owner.cpu().numpy().reshape(topk.shape)
        checked = audit(topk, seed_np, actual, support_np, expected)
        stats = planner.stats.cpu().numpy().tolist()
        predicted = sum(v[3] for v in stats)
        if a.mode != 'local':
            assert predicted == checked['saved']
        else:
            gains = [None] * 8
            dist.all_gather_object(gains, predicted)
            assert sum(gains) == checked['saved']
        for _ in range(a.warmup):
            runner()
        torch.cuda.synchronize()
        wall_ms, gpu_ms = [], []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(a.reps):
            if distributed:
                dist.barrier()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start.record()
            runner()
            end.record()
            end.synchronize()
            wall_ms.append(1000 * (time.perf_counter() - begin))
            gpu_ms.append(start.elapsed_time(end))
        # Rank-max per synchronized repetition is the distributed critical path.
        if distributed:
            timings = [None] * 8
            dist.all_gather_object(timings, {'wall_ms': wall_ms, 'gpu_ms': gpu_ms})
            rank_wall = np.asarray([t['wall_ms'] for t in timings])
            rank_gpu = np.asarray([t['gpu_ms'] for t in timings])
            wall_ms = rank_wall.max(axis=0).tolist()
            gpu_ms = rank_gpu.max(axis=0).tolist()
        else:
            timings = [{'wall_ms': wall_ms, 'gpu_ms': gpu_ms}]
        # Confirm repeated executions still produce the same owner.
        final_slice = planner.owner.cpu().numpy()
        expected_slice = expected[rank] if distributed else expected.reshape(-1, 8)
        assert np.array_equal(final_slice, expected_slice)
        row = dict(case=case, shape=list(topk.shape), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   audit=checked, median_gpu_ms=statistics.median(gpu_ms),
                   p95_gpu_ms=float(np.percentile(gpu_ms, 95)), median_wall_ms=statistics.median(wall_ms),
                   p95_wall_ms=float(np.percentile(wall_ms, 95)), rankmax_gpu_ms=gpu_ms,
                   rankmax_wall_ms=wall_ms, rank_timings=timings, stats_rank0=stats,
                   sources=planner.source_manifest())
        rows.append(row)
        if rank == 0:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps({'metadata': metadata, 'rows': rows}, indent=2) + '\n')
            print(json.dumps({'case': case, 'mode': a.mode, 'gpu_ms': row['median_gpu_ms'],
                              'wall_ms': row['median_wall_ms'], 'saved_pct': checked['saved_percent'],
                              'cpu_exact': True}), flush=True)
        del runner, graph, planner, seed, routes, support
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
