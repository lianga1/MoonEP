"""Two-rank residency component timing and Torch Profiler diagnostics.

Run after the unprofiled main benchmark, with exclusive GPU access:
  torchrun --standalone --nproc_per_node=2 experiments/two_gpu_overlap/profile_residency.py \
    --output outputs/two_gpu_overlap/residency_profile_attempt01

Does not modify the verified kernels. Reads prepared layers 0 and 7 by default.
This excludes native owner decoding and expert/token data communication. Both
ranks retain their raw eager and Graph Chrome traces. Diagnostic Graph samples
are kept separately from profiler durations and the accepted main benchmark.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from residency_cuda import TwoRankResidency

ROOT = Path(__file__).resolve().parents[2]


def summarize_trace(path):
    trace = json.loads(path.read_text())
    events = trace['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    if not kernels:
        raise RuntimeError(f'No CUDA kernel events in {path}; capture is unusable')
    grouped = defaultdict(lambda: dict(count=0, total_us=0.0, streams=set(), devices=set()))
    for event in kernels:
        row = grouped[event['name']]
        row['count'] += 1
        row['total_us'] += float(event['dur'])
        row['streams'].add(str(event.get('args', {}).get('stream', event.get('tid', 'unknown'))))
        row['devices'].add(str(event.get('args', {}).get('device', event.get('pid', 'unknown'))))
    rows = []
    duration_sum = sum(float(e['dur']) for e in kernels)
    for name, row in grouped.items():
        rows.append(dict(name=name, count=row['count'], total_us=row['total_us'],
                         mean_us=row['total_us']/row['count'],
                         fraction_of_kernel_sum=row['total_us']/duration_sum,
                         streams=sorted(row['streams']), devices=sorted(row['devices'])))
    rows.sort(key=lambda r: -r['total_us'])
    intervals = sorted((float(e['ts']), float(e['ts'])+float(e['dur'])) for e in kernels)
    begin, end = intervals[0]
    union = 0.0
    for left, right in intervals[1:]:
        if left > end:
            union += end-begin
            begin, end = left, right
        else:
            end = max(end, right)
    union += end-begin
    memory = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset') and e.get('ph') == 'X']
    return dict(kernel_events=len(kernels), kernel_duration_sum_us=duration_sum,
                kernel_busy_union_us=union,
                kernel_envelope_us=max(right for _, right in intervals)-intervals[0][0],
                gpu_memory_event_count=len(memory),
                gpu_memory_duration_sum_us=sum(float(e['dur']) for e in memory),
                kernels=rows,
                warning='Duration shares are attribution, not additive removable latency or a speedup prediction.')


def collect(value):
    result = [None]*dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def graph_timing(graph, repeats):
    dist.barrier()
    torch.cuda.synchronize()
    start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    tick = time.perf_counter()
    for _ in range(repeats):
        graph.replay()
    finish.record()
    finish.synchronize()
    return dict(cuda_ms=start.elapsed_time(finish)/repeats,
                host_ms=(time.perf_counter()-tick)*1000/repeats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=ROOT/'outputs/two_gpu_overlap/inputs')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 7])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--timing-repeats', type=int, default=10)
    parser.add_argument('--profile-repeats', type=int, default=3)
    parser.add_argument('--profile-modes', nargs='+', choices=['eager', 'graph'], default=['eager', 'graph'])
    args = parser.parse_args()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=timedelta(minutes=12))
    assert dist.get_world_size() == 2
    exists = [bool((args.output/'summary.json').exists()) if rank == 0 else None]
    dist.broadcast_object_list(exists, src=0)
    if exists[0]:
        raise FileExistsError('Choose a new output directory; prior diagnostic evidence is retained')
    args.output.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(local_rank)
    devices = collect(dict(rank=rank, gpu=torch.cuda.get_device_name(local_rank),
                           uuid=str(getattr(properties, 'uuid', 'unavailable')),
                           capability=list(torch.cuda.get_device_capability(local_rank))))
    report = dict(status='running', metadata=dict(created_at=datetime.now(timezone.utc).isoformat(),
                  torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                  devices=devices, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  scope='residency only; native owner decode is excluded',
                  timing_protocol='unprofiled Graph component samples; max rank CUDA event time per sample; separate diagnostic cohort',
                  profiler_protocol='CPU+CUDA eager mapping and warmed Graph traces; both physical ranks retained',
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__),Path(__file__).with_name('residency_cuda.py'),Path(__file__).with_name('residency_cuda.cu'))}),
                  layers=[], failures=[])
    def save():
        if rank == 0:
            (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    graphs = {}
    try:
        for layer in args.layers:
            raw_path = args.input_dir/f'layer_{layer:02d}.npz'
            raw = np.load(raw_path, allow_pickle=False)
            S = raw['topk'].shape[1]
            def gpu(x, dtype=torch.int32):
                return torch.tensor(x, device='cuda', dtype=dtype)
            routes, owner = gpu(raw['topk'][rank]), gpu(raw['owner'][rank])
            native_etc = gpu(raw['native_experts_to_copy'])
            residency = TwoRankResidency(gpu(raw['sample_routes']), gpu(raw['sample_ids'], torch.int64),
                                         local_tokens=S, rank=rank)
            residency.run(native_etc, owner, local_routes=routes)
            torch.cuda.synchronize()
            assert np.array_equal(residency.support.cpu().numpy(), raw['support'])
            functions = {
                'select_only': lambda: residency.select(native_etc),
                'samples_only': lambda: residency.gather_samples(routes, owner),
                'samples_select': lambda: residency.run(native_etc, owner, local_routes=routes),
            }
            row = dict(layer=layer, shape=list(raw['topk'].shape), sample_count=len(raw['sample_ids']),
                       input_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                       graph_timings={}, profiles=[], exact_support_before=True)
            report['layers'].append(row)
            for name, fn in functions.items():
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize();dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graphs[name] = graph
                graph.replay()
                torch.cuda.synchronize();dist.barrier()
                row['graph_timings'][name] = []
            names = list(functions)
            for sample in range(args.samples):
                order = names[sample%len(names):]+names[:sample%len(names)]
                for name in order:
                    ranks = collect(graph_timing(graphs[name], args.timing_repeats))
                    row['graph_timings'][name].append(dict(sample=sample, per_rank=ranks,
                          max_cuda_ms=max(r['cuda_ms'] for r in ranks),
                          max_host_ms=max(r['host_ms'] for r in ranks)))
                save()
            row['graph_median_ms'] = {name:statistics.median(x['max_cuda_ms'] for x in rounds)
                                      for name,rounds in row['graph_timings'].items()}
            # Profile only after all unprofiled component timings are retained.
            for mode in args.profile_modes:
                for name in ('select_only', 'samples_select'):
                    fn = functions[name] if mode == 'eager' else graphs[name].replay
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize();dist.barrier()
                    trace = args.output/f'layer{layer:02d}_rank{rank}_{name}_{mode}.trace.json'
                    failure = None
                    local_report = None
                    try:
                        with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as profiler:
                            for repeat in range(args.profile_repeats):
                                with torch.profiler.record_function(f'layer{layer}/rank{rank}/{name}/{mode}/repeat{repeat}'):
                                    fn()
                            torch.cuda.synchronize()
                        profiler.export_chrome_trace(str(trace))
                        local_report = dict(rank=rank, scope=name, mode=mode, trace=trace.name,
                                            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                                            **summarize_trace(trace))
                    except BaseException as exc:
                        failure = dict(layer=layer, rank=rank, scope=name, mode=mode,
                                       error=repr(exc), traceback=traceback.format_exc())
                    failures = [e for e in collect(failure) if e is not None]
                    report['failures'].extend(failures)
                    rows = collect(local_report)
                    row['profiles'].extend(r for r in rows if r is not None)
                    save()
            assert np.array_equal(residency.support.cpu().numpy(), raw['support'])
            row['exact_support_after'] = True
            for graph in graphs.values():
                graph.reset()
            graphs.clear();graph=None;fn=None
            torch.cuda.synchronize();dist.barrier()
            save()
            if rank == 0:
                print(json.dumps({'layer':layer,'graph_median_ms':row['graph_median_ms'],
                                  'profile_rows':len(row['profiles'])}), flush=True)
        report['status'] = 'failed' if report['failures'] else 'complete'
        save()
        if report['failures']:
            raise RuntimeError('Some profiler captures failed; retained separately in summary.json')
    except BaseException as exc:
        report['status'] = 'failed'
        report['failures'].append(dict(rank=rank, error=repr(exc), traceback=traceback.format_exc()))
        save()
        raise
    finally:
        for graph in graphs.values():
            graph.reset()
        graphs.clear();graph=None
        torch.cuda.synchronize()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
