"""True eight-physical-GPU planner benchmark on source-local router fixtures.

torchrun --standalone --nproc_per_node=8 -m tests.benchmark_sampled_group \
  --fixture-root /path/to/kernel_inputs --output /path/to/attempt01.json

Only planner execution is timed. No expert/token payload, GEMM or optimizer is
included. Fixture owners/support never enter the measured planner. Each process
reads only its own route shard; full routes are gathered to rank zero only for
untimed independent audits. Previous output attempts are never overwritten.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import socket
import statistics
import time
import traceback
import zipfile

import numpy as np
import torch
import torch.distributed as dist

from moonep import Buffer
from moonep.planning import allocate_planning_outputs, launch_moon_planning, launch_planning
from moonep.inter_rank_sync import launch_inter_rank_sync


ROOT = Path(__file__).resolve().parents[1]
R, S, E, K = 8, 16768, 352, 8
FIELDS = ('dst', 'experts_to_copy', 'zero_fill_ranges', 'remote_stats')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def read_local_routes(path, rank):
    # NPZ compression requires streaming preceding bytes, but no other rank's
    # routes are materialized as a numpy/Torch tensor in this process.
    with zipfile.ZipFile(path) as archive, archive.open('topk.npy') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f'Unsupported NPY header {version}')
        if shape != (R, S, K) or fortran or dtype != np.dtype('int32'):
            raise ValueError(f'Expected C-contiguous int32 {(R,S,K)}, got {shape}/{dtype}')
        nbytes = S*K*4
        remaining = rank*nbytes
        while remaining:
            skipped = f.read(min(remaining, 1024*1024))
            if not skipped:
                raise ValueError('Truncated router fixture')
            remaining -= len(skipped)
        data = f.read(nbytes)
        if len(data) != nbytes:
            raise ValueError('Truncated local route shard')
    routes = np.frombuffer(data, dtype=np.int32).reshape(S, K).copy()
    assert routes.min() >= 0 and routes.max() < E
    assert (np.diff(np.sort(routes, axis=1), axis=1) != 0).all()
    return routes


def precompile_planner(ctx):
    """Compile without launching a rank barrier, then synchronize host ranks."""
    from moonep import planning, inter_rank_sync
    c = ctx
    planning._get_compiled(*(int(c[k]) for k in (
        'R', 'E', 'B', 'S', 'K', 'NvS_capacity', 'NvS', 'num_vblocks',
        'meta_chunk_padded', 'TPE_OFF', 'PLAN_OFF', 'BARRIER_OFF', 'TOPK0_OFF',
        'ORDER_OFF', 'ORDER0_OFF', 'token_padding', 'num_sms')))
    inter_rank_sync._get_compiled(int(c['R']), int(c['meta_chunk_padded']),
        inter_rank_sync._num_threads_for_ranks(int(c['R'])), int(c['device']))
    torch.cuda.synchronize()
    dist.barrier()


def gather_root(value):
    gathered = [None]*R if dist.get_rank() == 0 else None
    dist.gather_object(value, gathered, dst=0)
    return gathered


def plan_snapshot(plan, cu, ctx, *, adapter=None):
    torch.cuda.synchronize()
    item = {name: getattr(plan, name).cpu().numpy().copy() for name in FIELDS}
    item.update(cu=cu.cpu().numpy().copy(), NvS=int(ctx['NvS']))
    if adapter is not None:
        item.update(status=int(adapter.status.item()),
            owner=adapter.owner.cpu().numpy().copy(),
            support=adapter.support.cpu().numpy().astype(bool),
            native_seed={name: getattr(adapter.seed, name).cpu().numpy().copy() for name in FIELDS},
            native_seed_cu=adapter.seed_cu.cpu().numpy().copy())
    return item


def audit_plans(items, routes, slots, optimized=False):
    """CPU audit from actual output addresses; no fixture oracle owner needed."""
    checks = {}
    def require(ok, label):
        checks[label] = bool(ok)
        if not ok:
            raise AssertionError(label)

    require(len({x['NvS'] for x in items}) == 1, 'same_address_stride')
    nvs = items[0]['NvS']
    raw = np.stack([x['dst'] for x in items]).reshape(R, S, K).astype(np.int64)
    address = np.where(raw < 0, -raw-1, raw)
    owner, offset = address//nvs, address % nvs
    require(((owner >= 0) & (owner < R)).all(), 'owner_in_range')
    require(np.unique(address).size == R*S*K, 'unique_address_for_every_compute_pair')
    loads = np.bincount(owner.ravel(), minlength=R)
    require(np.array_equal(loads, np.full(R, S*K)), 'exact_equal_compute_pair_load')
    seen = np.zeros((R, S), dtype=np.int32)
    for k in range(K):
        mask = 1 << owner[:, :, k]
        require(np.array_equal(raw[:, :, k] < 0, (seen & mask) != 0), f'dedup_sign_slot{k}')
        seen |= mask
    dispatch = np.zeros((R, R), dtype=np.int64)
    for source in range(R):
        for dest in range(R):
            if source != dest:
                dispatch[source, dest] = np.count_nonzero((owner[source] == dest).any(axis=1))
    copies = items[0]['experts_to_copy']
    require(copies.shape == (R, slots), 'copy_table_shape')
    require(all(np.array_equal(x['experts_to_copy'], copies) for x in items), 'same_copy_table_on_all_ranks')
    group = np.full((R, E), -1, dtype=np.int64)
    copy_matrix = np.zeros((R, R), dtype=np.int64)
    for rank in range(R):
        group[rank, rank*44:(rank+1)*44] = np.arange(rank*44, (rank+1)*44)
        active_slots = np.flatnonzero(copies[rank] >= 0)
        ids = copies[rank, active_slots]
        require(len(np.unique(ids)) == len(ids), f'rank{rank}_copies_unique')
        require(((ids >= 0) & (ids < E) & (ids//44 != rank)).all(), f'rank{rank}_copies_are_remote_experts')
        group[rank, ids] = E+active_slots
        for expert in ids:
            copy_matrix[int(expert)//44, rank] += 1
    assigned_group = group[owner, routes]
    require((assigned_group >= 0).all(), 'every_assigned_expert_present_in_physical_layout')
    ends = np.stack([x['cu'] for x in items]).astype(np.int64)
    starts = np.concatenate([np.zeros((R, 1), np.int64), ends[:, :-1]], axis=1)
    require(((ends >= starts) & (ends <= nvs)).all(), 'valid_cumulative_expert_segments')
    count = np.bincount((owner*E+routes).ravel(), minlength=R*E).reshape(R, E)
    row_start, row_end = starts[owner, assigned_group], ends[owner, assigned_group]
    require(((offset >= row_start) & (offset < row_end)).all(), 'address_in_correct_expert_segment')
    require((offset < row_start+count[owner, routes]).all(), 'no_compute_pair_in_padding')
    require(np.array_equal(count.sum(0), np.bincount(routes.ravel(), minlength=E)), 'expert_task_conservation')
    result = dict(checks=checks, passed=True, loads=loads.tolist(),
        remote_token_destinations=int(dispatch.sum()), dispatch_matrix=dispatch.tolist(),
        expert_copies=int((copies >= 0).sum()), expert_copies_by_rank=(copies >= 0).sum(1).tolist(),
        home_to_replica_matrix=copy_matrix.tolist(),
        max_endpoint_remote_tokens=int(max(dispatch.sum(0).max(), dispatch.sum(1).max())),
        owner_sha256=hashlib.sha256(owner.astype(np.int32).tobytes()).hexdigest())
    if optimized:
        status = [x['status'] for x in items]
        require(len(set(status)) == 1, 'same_global_status_on_all_ranks')
        fallback = status[0] != 0
        result.update(status_by_rank=status, fell_back_to_native=fallback,
                      optimizer_gain_claim_allowed=not fallback)
        if fallback:
            require(all(all(np.array_equal(x[name], x['native_seed'][name]) for name in FIELDS)
                        and np.array_equal(x['cu'], x['native_seed_cu']) for x in items),
                    'fallback_preserves_all_seed_outputs')
        else:
            support = items[0]['support']
            require(all(np.array_equal(x['support'], support) for x in items), 'same_support_on_all_ranks')
            require(support.shape == (R, E) and support[owner, routes].all(), 'optimizer_support_legal')
            for rank in range(R):
                remote = np.arange(E)//44 != rank
                require(support[rank, ~remote].all() and support[rank, remote].sum() <= slots,
                        f'rank{rank}_support_budget')
                require(np.array_equal(items[rank]['owner'].reshape(S, K), owner[rank]), f'rank{rank}_published_owner_equals_exchange')
    return result


def measure(fn, samples, replays):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    raw = []
    try:
        with torch.cuda.graph(graph):
            fn()
        graph.replay(); torch.cuda.synchronize(); dist.barrier()
        for sample in range(samples):
            dist.barrier(); torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter(); start.record()
            for _ in range(replays):
                graph.replay()
            end.record(); end.synchronize()
            per_rank = gather_root(dict(rank=dist.get_rank(),
                cuda_ms=start.elapsed_time(end)/replays,
                host_ms=(time.perf_counter()-host_start)*1000/replays))
            if dist.get_rank() == 0:
                raw.append(dict(sample=sample, replays=replays, per_rank=per_rank,
                    max_rank_cuda_ms=max(x['cuda_ms'] for x in per_rank),
                    max_rank_host_ms=max(x['host_ms'] for x in per_rank)))
    finally:
        torch.cuda.synchronize()
        graph.reset()
    if dist.get_rank() != 0:
        return None
    medians = [dict(rank=rank,
        cuda_ms=statistics.median(x['per_rank'][rank]['cuda_ms'] for x in raw),
        host_ms=statistics.median(x['per_rank'][rank]['host_ms'] for x in raw)) for rank in range(R)]
    return dict(raw=raw, per_rank_medians=medians,
        max_rank_median_cuda_ms=max(x['cuda_ms'] for x in medians),
        max_rank_median_host_ms=max(x['host_ms'] for x in medians),
        median_sample_max_rank_cuda_ms=statistics.median(x['max_rank_cuda_ms'] for x in raw))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--layers', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--replays-per-sample', type=int, default=5)
    ap.add_argument('--slots', type=int, choices=[8, 44], default=8)
    ap.add_argument('--hidden', type=int, default=256)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous attempts; use a fresh output name')
    if args.samples < 1 or args.replays_per_sample < 1:
        raise ValueError('samples and replays must be positive')
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    assert dist.get_world_size() == R and int(os.environ['LOCAL_WORLD_SIZE']) == R
    assert torch.cuda.device_count() >= R
    identities = [None]*R
    dist.all_gather_object(identities, (socket.gethostname(), torch.cuda.current_device()))
    assert len(set(identities)) == R
    record = dict(status='running', metadata=dict(
        physical_ranks=R, original_trace_R=R, S=S, N=R*S, E=E, K=K, B=args.slots,
        hidden_allocated_but_not_transferred=args.hidden, rounds=4, sample_seed=20261026,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(), samples=args.samples, replays_per_sample=args.replays_per_sample,
        timing='CUDA Graph; per-sample mean of five or configured replays; primary=max of rank medians; sample maxima also retained',
        included=['common native pre-planning rank sync', 'native seed planner', 'live sample collective',
                  'GPU residency', 'four live count collectives', 'source-local exchange',
                  'physical materialization', 'global status reduction', 'fallback selection', 'source record publication'],
        excluded=['fixture load', 'tpe histogram', 'JIT', 'audits', 'dedup-map builder in token dispatch',
                  'expert weight transport', 'token dispatch/combine', 'gradient payload', 'GEMM', 'optimizer'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__)]+[x for x in (ROOT/'moonep').rglob('*') if x.is_file() and x.suffix in ('.py','.cu','.cuh')]}),
        layers=[], failures=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        if rank == 0:
            args.output.write_text(json.dumps(record, indent=2)+'\n')
    buffer = None
    save()
    try:
        buffer = Buffer(S, args.hidden, K, E, R, B=args.slots, enable_pdl=False,
                        planning_backend='sampled_group', planning_rounds=4,
                        planning_sample_seed=20261026, explicitly_destroy=True)
        ctx = buffer._require_ctx(); precompile_planner(ctx)
        adapter = ctx['sampled_group_planner']
        for layer in map(int, args.layers.split(',')):
            file = args.fixture_root/f'layer_{layer:02}.npz'
            local_routes = read_local_routes(file, rank)
            routes = torch.from_numpy(local_routes).to('cuda')
            tpe = torch.bincount(routes.flatten().long(), minlength=E).int()
            audit_routes = gather_root(local_routes)
            baseline, base_cu = allocate_planning_outputs(ctx)
            optimized, opt_cu = allocate_planning_outputs(ctx)
            def moon():
                launch_inter_rank_sync(ctx)
                launch_moon_planning(ctx, routes.flatten(), tpe, base_cu, baseline)
            def sampled():
                launch_inter_rank_sync(ctx)
                launch_planning(ctx, routes.flatten(), tpe, opt_cu, optimized)
            row = dict(layer=layer, fixture=dict(name=file.name, sha256=sha(file)), variants={})
            if rank == 0:
                record['layers'].append(row); save()
            for name, fn in (('moon', moon), ('sampled_group', sampled)):
                result = measure(fn, args.samples, args.replays_per_sample)
                if rank == 0:
                    row['variants'][name] = result; save()
            # Restore both current-layer plans AFTER both graphs are reset.
            # Sampled must be last: its seed/support/owner scratch is audited.
            moon(); torch.cuda.synchronize()
            base_items = gather_root(plan_snapshot(baseline, base_cu, ctx))
            sampled(); torch.cuda.synchronize()
            opt_items = gather_root(plan_snapshot(optimized, opt_cu, ctx, adapter=adapter))
            route_checks = gather_root(bool(torch.equal(adapter.routes, routes)))
            check = [None]
            if rank == 0:
                row['planner_status_by_rank'] = [x['status'] for x in opt_items]
                try:
                    assert all(route_checks), 'backend reused stale routes from an earlier layer'
                    assert all(all(np.array_equal(b[name], o['native_seed'][name]) for name in FIELDS)
                               and np.array_equal(b['cu'], o['native_seed_cu'])
                               for b, o in zip(base_items, opt_items)), 'sampled native seed differs from direct native baseline'
                    full_routes = np.stack(audit_routes)
                    a = audit_plans(base_items, full_routes, args.slots)
                    b = audit_plans(opt_items, full_routes, args.slots, optimized=True)
                    assert b['remote_token_destinations'] <= a['remote_token_destinations']
                    if b['fell_back_to_native']:
                        assert b['remote_token_destinations'] == a['remote_token_destinations']
                    row.update(audit=dict(native=a, sampled_group=b),
                        saved_remote_token_destinations=a['remote_token_destinations']-b['remote_token_destinations'],
                        additional_expert_copies=b['expert_copies']-a['expert_copies'],
                        additional_planner_ms=row['variants']['sampled_group']['max_rank_median_cuda_ms']-
                                              row['variants']['moon']['max_rank_median_cuda_ms'],
                        runtime_routes_match_local_fixture=True, native_seed_matches_direct_baseline=True,
                        post_timing_audit_passed=True)
                    check[0] = dict(passed=True)
                except Exception:
                    row['post_timing_audit_passed'] = False
                    check[0] = dict(passed=False, traceback=traceback.format_exc())
                    row['audit_failure'] = check[0]['traceback']
                save()
            dist.broadcast_object_list(check, src=0)
            if not check[0]['passed']:
                raise AssertionError(check[0]['traceback'])
            dist.barrier()
        if rank == 0:
            record['summary'] = dict(
                native_planner_sum_layer_ms=sum(x['variants']['moon']['max_rank_median_cuda_ms'] for x in record['layers']),
                sampled_planner_sum_layer_ms=sum(x['variants']['sampled_group']['max_rank_median_cuda_ms'] for x in record['layers']),
                fallback_layers=[x['layer'] for x in record['layers'] if x['audit']['sampled_group']['fell_back_to_native']],
                all_post_timing_audits_passed=all(x['post_timing_audit_passed'] for x in record['layers']))
            record['status'] = 'complete'; save()
    except Exception:
        failure = dict(rank=rank, traceback=traceback.format_exc())
        failpath = args.output.with_name(args.output.stem+f'_rank{rank}_failure.json')
        failpath.write_text(json.dumps(failure, indent=2)+'\n')
        if rank == 0:
            record['status'] = 'failed'; record['failures'].append(failure); save()
        raise
    finally:
        if buffer is not None:
            torch.cuda.synchronize(); buffer.destroy()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
