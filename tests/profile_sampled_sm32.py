"""Eight-rank bounded-planner stage timing, with three separate evidence lanes.

  torchrun --standalone --nproc_per_node=8 -m tests.profile_sampled_sm32 \
    --fixture-root /path/to/kernel_inputs --output /fresh/unfused32.json
  # Add --fuse for the fused path. The default workload is cap128/four rounds.

1. Production full-planner CUDA Graph timing is uninstrumented (unchanged
   benchmark helper), before and after all diagnostic instrumentation.
2. Explicit external CUDA events surround every Python/extension-call phase.
   Two untimed production Graphs align ranks immediately before each sample.
3. Eager correlation maps phases to individual GPU launches. A production
   Graph is profiled for 2 discarded + 5 retained replays, without any host
   collective between them. All raw trace events and replay durations survive.

Host events cannot split the internals of a fused peer kernel or native seed.
Those remain explicitly mixed compute/communication/wait. A barrier kernel's
duration includes fences/signaling and arrival skew, not only spin waiting.
No production source, CUDA kernel, planner inputs or launch order is changed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import statistics
import traceback
import uuid

import numpy as np
import torch
import torch.distributed as dist

from moonep import Buffer, SampledPlannerConfig
from moonep.inter_rank_sync import launch_inter_rank_sync
from moonep.planning import allocate_planning_outputs, launch_moon_planning, launch_planning
from tests.benchmark_sampled_group import (
    E, FIELDS, K, R, S, ROOT, audit_plans, gather_root, measure,
    precompile_planner, read_local_routes, sha,
)
from tests.profile_sampled_stages import (
    broadcast_outcome, collective_local_check, complete_snapshot,
    compare_snapshots, export_trace, output_hashes,
)
from tests.profile_sampled_sm32_trace import eager_mapping, load_trace, summarize_graph


class BoundedStages:
    """Literal expansion of the production run into host-call boundaries.

    Individual C++ wrapper subkernels are resolved by profiler correlations,
    not by rewriting the wrapper or timing a different synthetic workload.
    """
    def __init__(self, ctx, adapter, topk, tpe, cu, plan):
        a, rt = adapter, adapter.runtime
        res, ex, mat = a.residency, a.exchange, a.materializer
        if rt is None:
            raise ValueError('This harness is only for a positive SM limit')
        self.stages, self.events = [], None

        def add(name, kind, fn, note=''):
            self.stages.append(dict(index=len(self.stages), name=name, kind=kind, fn=fn, note=note))

        add('pre_sync', 'peer_barrier', lambda: launch_inter_rank_sync(ctx))
        add('input_copy', 'local', lambda: rt.copy(topk, a.routes))
        add('native_seed', 'mixed', lambda: launch_moon_planning(
            a.planning_ctx, topk, tpe, a.seed_cu, a.seed),
            'Original native seed includes its own peer metadata and synchronization.')
        add('decode_owner', 'local', lambda: mat.decode_owner(a.seed.dst, a.seed_owner))
        add('residency_gather_local', 'local', lambda: res.gather_local(
            a.routes, a.seed_owner, a.seed.experts_to_copy))
        add('residency_sample_collective', 'peer_collective', lambda: rt.all_reduce(res.packet, 'sum'))
        # Old 802af3f builds lack this private eligibility flag and retain the
        # original single-host-call phase. New supported builds split records,
        # distributed frequency counting, and candidate selection exactly as
        # production select() does; the new all-gather must remain visible as
        # communication rather than being charged to local residency work.
        self.frequency_sharding_enabled = bool(getattr(res, 'frequency_sharding_enabled', False))
        if self.frequency_sharding_enabled:
            add('residency_records_prepare', 'local', lambda: res.select_prepare(a.seed.experts_to_copy),
                'Mandatory validation and construction of the same complete sampled records on every rank.')
            add('residency_frequency_shard', 'local', res.select_frequency_shard,
                'Compute this rank\'s output records; each query still compares against its complete sample group.')
            add('residency_frequency_collective', 'peer_collective',
                lambda: res.collective.all_gather(res.frequency_packed, res.frequency_gathered),
                'Live all-gather of rank-sharded frequency outputs; included in the planner.')
            add('residency_frequency_unpack', 'local', res.select_frequency_unpack,
                'Restore full frequency order from the rank-major gathered shards.')
            add('residency_candidate_finalize', 'local', lambda: res.select_finalize(a.seed.experts_to_copy),
                'Top-g selection, candidate creation/pruning/sorting, and greedy expert residency selection.')
        else:
            add('residency_select', 'local', lambda: res.select(a.seed.experts_to_copy),
                'Profiler resolves all validation/records/frequency/top-g/sort/greedy launches.')
        add('materialize_support', 'local', lambda: mat.prepare_support(a.support))
        add('exchange_prepare', 'local', ex.prepare_support)
        add('exchange_reset', 'local', lambda: ex.reset(a.seed_owner))
        for step in range(a.rounds):
            add(f'round{step+1}_propose_ordinal_prefix', 'local', lambda step=step: ex.propose(step),
                'Profiler resolves propose, stable ordinal and block prefix. Selective production keeps proposal/ordinal separate; only a private diagnostic override joins them.')
            add(f'round{step+1}_counts_collective', 'peer_collective',
                lambda: rt.all_gather(ex.counts, a.gathered_counts.view(-1)))
            add(f'round{step+1}_quota', 'local', lambda step=step: ex.quota(step, a.gathered_counts))
            add(f'round{step+1}_apply', 'local', ex.apply)
        add('materialize_prepare', 'local', lambda: mat.ext.prepare_bounded(
            a.routes, a.owner, a.seed_owner, mat.native_order, mat.lut,
            mat.keys_in, mat.keys_out, mat.ids_in, mat.ids_out, mat.starts,
            mat.counts, mat.sort_counts, mat.sort_prefix, mat.duplicates, mat.status,
            mat.R, mat.E, mat.K, mat.sm_limit, mat.fuse),
            'Profiler resolves clear, metadata/input/histogram (possibly fused), histogram prefix, scatter and starts.')
        add('materialize_counts_collective', 'peer_collective',
            lambda: rt.all_gather(mat.counts.view(-1), mat.gathered_counts.view(-1)))
        add('materialize_layout_encode', 'local', lambda: mat.ext.layout_encode_bounded(
            a.routes, a.owner, mat.ids_out, mat.starts, mat.gathered_counts,
            mat.lut, mat.copies, mat.offsets, mat.prefix, mat.totals, mat.duplicates,
            mat.plan.dst, mat.cu, mat.plan.zero_fill_ranges, mat.plan.remote_stats,
            mat.status, mat.R, mat.E, mat.B, mat.K, mat.rank, mat.padding, mat.NvS, mat.sm_limit))
        add('finish_status_merge', 'local', lambda: rt.ext.status(
            res.status, mat.status, a.force_reject, a.status))
        add('finish_status_collective', 'peer_collective', lambda: rt.all_reduce(a.status, 'max'))
        add('finish_select', 'local', lambda: rt.ext.select_all(a.status, a.valid,
            [mat.cu]+[getattr(mat.plan, n) for n in FIELDS],
            [a.seed_cu]+[getattr(a.seed, n) for n in FIELDS],
            [cu]+[getattr(plan, n) for n in FIELDS], rt.limit, rt.fuse))
        stride, offset = int(ctx['meta_chunk_padded']), int(ctx['SRC_INFO_OFF'])
        add('publication_clear', 'local', lambda: mat.ext.fill_bounded(
            ctx['meta_buf'], mat.rank*stride+offset, mat.NvS, -1, mat.sm_limit))
        add('publication_barrier_before', 'peer_barrier', lambda: launch_inter_rank_sync(ctx))
        add('publication_write', 'peer_metadata', lambda: mat.ext.publish_bounded(
            plan.dst, ctx['meta_buf'], mat.rank, mat.NvS, stride, offset, mat.sm_limit))
        add('publication_barrier_after', 'peer_barrier', lambda: launch_inter_rank_sync(ctx))

    def manifest(self):
        return [{k: v for k, v in stage.items() if k != 'fn'} for stage in self.stages]

    def warm_events(self):
        self.events = [torch.cuda.Event(enable_timing=True, external=True)
                       for _ in range(len(self.stages)+1)]
        for event in self.events:
            event.record()
        self.events[-1].synchronize()

    def run(self, *, events=False, labels=False):
        if events:
            self.events[0].record()
        for index, stage in enumerate(self.stages):
            if labels:
                with torch.profiler.record_function(f"sm32_stage/{index:02d}/{stage['name']}"):
                    stage['fn']()
            else:
                stage['fn']()
            if events:
                self.events[index+1].record()


def measure_stage_events(split, production, samples, alignment_replays):
    split.warm_events()
    for _ in range(3):
        production()
        split.run(events=True)
    torch.cuda.synchronize(); dist.barrier()
    graph, guard = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    raw = []
    try:
        with torch.cuda.graph(guard):
            production()
        with torch.cuda.graph(graph):
            split.run(events=True)
        guard.replay(); graph.replay(); torch.cuda.synchronize(); dist.barrier()
        for sample in range(samples):
            # The host barrier is outside events. Two full GPU planners then
            # absorb CPU entry skew before the measured external event nodes.
            dist.barrier(); torch.cuda.synchronize()
            for _ in range(alignment_replays):
                guard.replay()
            graph.replay()
            split.events[-1].synchronize()
            stage_us = {stage['name']: 1000*split.events[i].elapsed_time(split.events[i+1])
                        for i, stage in enumerate(split.stages)}
            total = 1000*split.events[0].elapsed_time(split.events[-1])
            raw.append(dict(sample=sample, stage_us=stage_us, total_us=total,
                sum_stage_us=sum(stage_us.values()), closure_error_us=total-sum(stage_us.values())))
    finally:
        torch.cuda.synchronize(); graph.reset(); guard.reset()
    return dict(rank=dist.get_rank(), raw=raw, alignment_replays=alignment_replays,
        median_total_us=statistics.median(x['total_us'] for x in raw),
        median_stage_us={s['name']: statistics.median(x['stage_us'][s['name']] for x in raw)
                         for s in split.stages},
        max_absolute_closure_error_us=max(abs(x['closure_error_us']) for x in raw),
        warning='Event nodes perturb timing; N+1 shared boundaries make each sample close, but sums of independently computed medians need not close.')


def profile_steady(split, production, directory, layer, discarded, retained, sm_limit):
    rank = dist.get_rank()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    for _ in range(3):
        split.run()
    torch.cuda.synchronize(); dist.barrier()
    # This eager pass maps launches only. Its durations are never reported.
    with torch.profiler.profile(activities=activities, record_shapes=False,
                                with_stack=False, profile_memory=False) as profiler:
        split.run(labels=True)
        torch.cuda.synchronize()
    eager_file = export_trace(profiler, directory, f'layer{layer:02d}_rank{rank}_eager_mapping')
    torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            production()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(); dist.barrier()
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    with_stack=False, profile_memory=False) as profiler:
            # Rank-local profiler initialization can differ substantially. It
            # contaminates the first GPU peer barrier; retain and discard two
            # complete executions, then measure the following steady replays.
            # Do not put dist.barrier, sync, CPU reads or trace export between.
            for replay in range(discarded+retained):
                with torch.profiler.record_function(f'sm32_graph/replay{replay:02d}'):
                    graph.replay()
            torch.cuda.synchronize()
        graph_file = export_trace(profiler, directory, f'layer{layer:02d}_rank{rank}_steady_graph')
    finally:
        torch.cuda.synchronize(); graph.reset()
    dist.barrier()
    mapping = eager_mapping(load_trace(eager_file['path']), split.manifest())
    summary = summarize_graph(load_trace(graph_file['path']), mapping, discarded, retained, sm_limit)
    return dict(rank=rank, eager_trace=eager_file, graph_trace=graph_file, mapping=mapping, **summary)


def source_manifest():
    files = [p for p in (ROOT/'moonep').rglob('*') if p.is_file()
             and p.suffix in ('.py', '.cu', '.cuh') and not p.name.startswith('._')]
    files.extend(Path(__file__).with_name(name) for name in (
        'profile_sampled_sm32.py', 'profile_sampled_sm32_trace.py',
        'profile_sampled_stages.py', 'benchmark_sampled_group.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--profile-dir', type=Path)
    ap.add_argument('--layers', default='0,7')
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--replays', type=int, default=5)
    ap.add_argument('--stage-samples', type=int, default=10)
    ap.add_argument('--alignment-replays', type=int, default=2)
    ap.add_argument('--trace-discard', type=int, default=2)
    ap.add_argument('--trace-replays', type=int, default=5)
    ap.add_argument('--sm-limit', type=int, default=32)
    ap.add_argument('--fuse', action='store_true')
    args = ap.parse_args()
    args.profile_dir = args.profile_dir or args.output.with_name(args.output.stem+'_traces')
    layers = [int(x) for x in args.layers.split(',')]
    if not layers or len(set(layers)) != len(layers) or any(x not in range(8) for x in layers):
        raise ValueError('layers must be unique indices in 0..7')
    if min(args.samples, args.replays, args.stage_samples, args.trace_replays, args.sm_limit) < 1:
        raise ValueError('sample/replay counts and SM limit must be positive')
    if min(args.alignment_replays, args.trace_discard) < 2:
        raise ValueError('At least two warm alignment/discard replays are required')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    buffer, record, reserved = None, None, False
    phase = 'initialization'
    rank_raw = dict(rank=rank, layers=[])
    rank_path = args.output.with_name(args.output.stem+f'_rank{rank}_raw.json')

    def save():
        if rank == 0 and reserved and record is not None:
            args.output.write_text(json.dumps(record, indent=2)+'\n')

    def save_local():
        rank_path.write_text(json.dumps(rank_raw, indent=2)+'\n')

    def root_check(fn):
        outcome = None
        if rank == 0:
            try:
                value = fn()
                outcome = dict(passed=True)
            except Exception:
                value = None
                outcome = dict(passed=False, error=traceback.format_exc())
        else:
            value = None
        broadcast_outcome(outcome)
        return value

    try:
        torch.cuda.set_device(local); torch.set_num_threads(1)
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
        if dist.get_world_size() != R or int(os.environ['LOCAL_WORLD_SIZE']) != R:
            raise ValueError('Eight physical local ranks are required')
        props = torch.cuda.get_device_properties(local)
        identities = gather_root(dict(rank=rank, host=socket.gethostname(), local_rank=local,
            uuid=str(props.uuid), sms=props.multi_processor_count))
        root_check(lambda: (len({x['uuid'] for x in identities}) == R and
            len({x['host'] for x in identities}) == 1) or (_ for _ in ()).throw(
                AssertionError('Expected eight different GPUs on one host')))
        outcome = None
        if rank == 0:
            try:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('x') as file:
                    file.write('{"status":"initializing"}\n')
                reserved = True
                args.profile_dir.mkdir(parents=True, exist_ok=False)
                outcome = dict(passed=True)
            except Exception:
                outcome = dict(passed=False, error=traceback.format_exc())
        broadcast_outcome(outcome)
        collective_local_check('reserve per-rank raw output', lambda: rank_path.touch(exist_ok=False))
        cfg = SampledPlannerConfig(candidate_limit=128, rounds=4, sm_limit=args.sm_limit, fuse=args.fuse)
        record = dict(status='running', metadata=dict(
            base_commit='802af3f', R=R, N=R*S, S=S, E=E, K=K, B=8,
            config=cfg.to_dict(), devices=identities, gpu=torch.cuda.get_device_name(),
            torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            source_files=source_manifest(),
            timing='Formal production Graph before/after; separate external-event phases; steady GPU trace after two discarded replays.',
            excluded=['allocation', 'JIT', 'histogram construction', 'audits', 'expert/token/gradient payload', 'GEMM', 'optimizer'],
            limitation='No host-based measurement can separate pure wait, local work and communication within native seed or fused peer kernels.'),
            layers=[], failures=[])
        save()
        phase = 'buffer_and_jit'
        buffer = Buffer(S, 256, K, E, R, B=8, enable_pdl=False,
            planning_backend='sampled_group', planner_config=cfg, explicitly_destroy=True)
        ctx = buffer._require_ctx(); adapter = ctx['sampled_group_planner']
        precompile_planner(ctx); precompile_planner(adapter.planning_ctx)
        plan, cu = allocate_planning_outputs(ctx)
        native_plan, native_cu = allocate_planning_outputs(ctx)
        for layer in layers:
            phase = f'layer{layer:02d}_input'
            path = args.fixture_root/f'layer_{layer:02d}.npz'
            local_routes = collective_local_check(phase, lambda: read_local_routes(path, rank))
            routes = torch.from_numpy(local_routes).to('cuda'); topk = routes.flatten()
            tpe = torch.bincount(topk.long(), minlength=E).int()
            all_routes = gather_root(local_routes)

            def production():
                launch_inter_rank_sync(ctx)
                launch_planning(ctx, topk, tpe, cu, plan)

            def native():
                launch_inter_rank_sync(ctx)
                launch_moon_planning(ctx, topk, tpe, native_cu, native_plan)

            split = BoundedStages(ctx, adapter, topk, tpe, cu, plan)
            row = dict(layer=layer, fixture_sha256=sha(path), stages=split.manifest(),
                       residency_frequency_sharding_enabled=split.frequency_sharding_enabled,
                       correctness={})
            record['layers'].append(row); save()
            local_row = dict(layer=layer); rank_raw['layers'].append(local_row); save_local()
            phase = f'layer{layer:02d}_formal_before'
            row['formal_native'] = measure(native, args.samples, args.replays)
            native_items = gather_root(complete_snapshot(native_plan, native_cu, ctx))
            row['formal_before'] = measure(production, args.samples, args.replays)
            expected = complete_snapshot(plan, cu, ctx, adapter)
            items = gather_root(expected)

            def audit():
                assert all(x['status'] == 0 for x in items), 'Unexpected fallback'
                result = dict(native=audit_plans(native_items, np.stack(all_routes), 8),
                    sampled=audit_plans(items, np.stack(all_routes), 8, optimized=True),
                    production_output_hashes_by_rank=[output_hashes(x) for x in items])
                assert result['sampled']['remote_token_destinations'] <= result['native']['remote_token_destinations']
                return result

            row['correctness'].update(root_check(audit) or {}); save()

            def check_current(label):
                actual = collective_local_check(label, lambda: complete_snapshot(plan, cu, ctx, adapter))
                comparison = gather_root(dict(rank=rank, mismatches=compare_snapshots(expected, actual)))
                def check():
                    assert all(not x['mismatches'] for x in comparison), comparison
                    return comparison
                row['correctness'][label] = root_check(check)
                save()

            split.run(); torch.cuda.synchronize(); check_current('split_eager_exact')
            phase = f'layer{layer:02d}_events'
            local_row['stage_events'] = measure_stage_events(
                split, production, args.stage_samples, args.alignment_replays)
            save_local()
            row['stage_events_by_rank'] = gather_root(local_row['stage_events'])
            check_current('external_event_graph_exact')
            phase = f'layer{layer:02d}_profiler'
            local_row['trace'] = collective_local_check(phase, lambda: profile_steady(
                split, production, args.profile_dir, layer, args.trace_discard, args.trace_replays, args.sm_limit))
            save_local(); row['traces_by_rank'] = gather_root(local_row['trace'])
            check_current('steady_profiler_graph_exact')
            phase = f'layer{layer:02d}_formal_after'
            row['formal_after'] = measure(production, args.samples, args.replays)
            check_current('formal_after_graph_exact')
            if rank == 0:
                before = row['formal_before']['max_rank_median_cuda_ms']*1000
                after = row['formal_after']['max_rank_median_cuda_ms']*1000
                events = max(x['median_total_us'] for x in row['stage_events_by_rank'])
                profiled = max(x['median_gpu_span_us'] for x in row['traces_by_rank'])
                row['timing_comparison'] = dict(formal_before_us=before, formal_after_us=after,
                    stage_event_total_us=events, trace_gpu_span_us=profiled,
                    event_minus_formal_before_us=events-before, trace_minus_formal_before_us=profiled-before,
                    warning='Different instrumentation and replay batching. Differences are observed perturbation, not a calibrated subtraction. Do not sum max-rank stage medians as a measured critical path.')
                print(json.dumps(dict(layer=layer, fuse=args.fuse, formal_before_us=before,
                    formal_after_us=after, events_us=events, trace_us=profiled)), flush=True)
            row['completed'] = True; save()
        record['status'] = 'complete'; save()
    except Exception:
        failure = dict(rank=rank, phase=phase, traceback=traceback.format_exc())
        failure_path = args.output.with_name(args.output.stem+f'_rank{rank}_{uuid.uuid4().hex}_failure.json')
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failure, indent=2)+'\n')
        if rank == 0 and record is not None:
            record['status'] = 'failed'; record['failures'].append(failure); save()
        raise
    finally:
        if buffer is not None:
            torch.cuda.synchronize(); buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
