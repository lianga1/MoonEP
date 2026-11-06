"""Eight-physical-GPU microbenchmark of COMPLETE bounded peer collectives.

torchrun --standalone --nproc_per_node=8 -m tests.benchmark_peer_grids \
  --output evidence/peer_grids_01.json --samples 20

Every measured Graph execution begins with TWO untimed full collectives for
device-side alignment. External CUDA events inside the graph bracket each
complete measured collective. Input restoration is outside those event pairs,
but separately included in a measured group interval. SUM/MAX are in-place and
restore the original rank-distinct input before every invocation, including
alignment calls. Input changes on EVERY replay; every output is checked.
fused_auto is the LEGACY all-fused control. Production selective fusion uses
the unfused row for payloads >256 words and fused_auto for smaller payloads.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import traceback

import numpy as np
import torch
import torch.distributed as dist

from moonep.sampled_runtime import BoundedPlannerRuntime

R, LIMIT = 8, 32
ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = (
    ('exchange_counts', 1152, 'gather', 4),
    ('materialize_counts', 2816, 'gather', 1),
    ('sample_packet', 4609, 'sum', 1),
    ('global_status', 1, 'max', 1),
)
VARIANTS = (('unfused', 0), ('fused_auto', -1), ('cooperative_1', 1),
            ('cooperative_2', 2), ('cooperative_4', 4), ('cooperative_8', 8),
            ('cooperative_16', 16), ('cooperative_32', 32))


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def inputs(phase, rank, rounds, words):
    # Negative values exercise MAX's INT_MIN identity. The bounded sample
    # count keeps even the exact SUM well within the signed int32 contract.
    call = np.arange(rounds, dtype=np.int64)[:, None]
    word = np.arange(words, dtype=np.int64)[None, :]
    return (phase * 1000 + rank * 10000 + call * 100 + word * 3 - 1000000).astype(np.int32)


def expected(phase, rounds, words, op):
    all_inputs = np.stack([inputs(phase, rank, rounds, words) for rank in range(R)])
    if op == 'gather':
        return all_inputs.transpose(1, 0, 2).reshape(rounds, R * words).copy()
    value = all_inputs.sum(0, dtype=np.int64) if op == 'sum' else all_inputs.max(0)
    assert value.min() >= -(2**31) and value.max() < 2**31
    return value.astype(np.int32)


class Workload:
    def __init__(self, spec):
        self.name, self.words, self.op, self.calls = spec
        self.seed = torch.zeros((self.calls, self.words), device='cuda', dtype=torch.int32)
        self.value = torch.empty_like(self.seed)
        if self.op == 'gather':
            self.output = torch.empty((self.calls, R * self.words), device='cuda', dtype=torch.int32)
            self.warm_output = torch.empty(R * self.words, device='cuda', dtype=torch.int32)
        else:
            self.output = self.value  # Deliberately exercise in-place SUM/MAX.
            self.warm_output = self.value[0]


def event():
    # external=True preserves explicit event nodes on every Graph replay;
    # ordinary dependency-only capture events cannot provide this timing.
    return torch.cuda.Event(enable_timing=True, external=True)


def capture(runtime, work, grid):
    starts = [event() for _ in range(work.calls)]
    ends = [event() for _ in range(work.calls)]
    group_start, group_end = event(), event()
    graph = torch.cuda.CUDAGraph()

    def invoke():
        for _ in range(2):
            runtime.copy(work.seed[0], work.value[0])
            runtime._diagnostic_collective(work.value[0], work.warm_output, op=work.op, grid=grid)
        group_start.record()
        for call in range(work.calls):
            # This copy is mandatory for repeated in-place SUM. It is the
            # same bounded kernel in every variant and is outside call timing.
            runtime.copy(work.seed[call], work.value[call])
            starts[call].record()
            runtime._diagnostic_collective(work.value[call], work.output[call], op=work.op, grid=grid)
            ends[call].record()
        group_end.record()

    # Load modules and materialize event handles before capture on every rank.
    # Setup/warmup collectives are outside the benchmark record.
    invoke(); torch.cuda.synchronize(); dist.barrier()
    with torch.cuda.graph(graph):
        invoke()
    dist.barrier()
    return dict(graph=graph, starts=starts, ends=ends, group_start=group_start, group_end=group_end)


def gather(value, rank):
    values = [None] * R if rank == 0 else None
    dist.gather_object(value, values, dst=0)
    return values


def summarize(records, samples):
    result = {}
    for name, words, op, calls in WORKLOADS:
        variants = {}
        for variant, grid in VARIANTS:
            rows = [x for x in records if x['workload'] == name and x['variant'] == variant]
            assert len(rows) == samples and sorted(x['sample'] for x in rows) == list(range(samples))
            ranks = []
            for rank in range(R):
                items = [row['per_rank'][rank] for row in rows]
                assert all(x['rank'] == rank and x['passed'] for x in items)
                ranks.append(dict(rank=rank,
                    median_total_collective_us=statistics.median(x['total_collective_us'] for x in items),
                    median_group_with_input_reset_us=statistics.median(x['group_with_input_reset_us'] for x in items),
                    median_by_call_us=[statistics.median(x['call_us'][call] for x in items) for call in range(calls)]))
            maximum = max(x['median_total_collective_us'] for x in ranks)
            variants[variant] = dict(grid=grid, per_rank_medians=ranks,
                max_rank_median_total_collective_us=maximum,
                max_rank_median_total_divided_by_calls_us=maximum / calls,
                max_rank_median_group_with_input_reset_us=max(x['median_group_with_input_reset_us'] for x in ranks))
        baseline = variants['unfused']['max_rank_median_total_collective_us']
        for value in variants.values():
            value['latency_change_vs_unfused_percent'] = 100 * (value['max_rank_median_total_collective_us'] / baseline - 1)
        result[name] = dict(words=words, op=op, calls=calls, variants=variants)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    assert 1 <= args.samples <= 256
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == R and int(os.environ['LOCAL_WORLD_SIZE']) == R
    torch.cuda.set_device(local); torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    runtime, record, captures = None, None, {}

    def save():
        if rank == 0 and record is not None:
            args.output.write_text(json.dumps(record, indent=2) + '\n')

    try:
        props = torch.cuda.get_device_properties(local)
        identities = gather(dict(rank=rank, local_device=local, host=socket.gethostname(),
                                 uuid=str(props.uuid), gpu=props.name, sms=props.multi_processor_count), rank)
        verdict = [True]
        if rank == 0:
            verdict[0] = (not args.output.exists() and len({x['uuid'] for x in identities}) == R and
                          len({x['host'] for x in identities}) == 1 and all(x['sms'] >= LIMIT for x in identities))
        dist.broadcast_object_list(verdict, src=0)
        assert verdict[0], 'need a new output path and eight distinct GPUs on one host, each with >=32 SMs'
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            record = dict(status='running', failures=[], samples=[], metadata=dict(
                ranks=R, identities=identities, samples=args.samples, sm_limit=LIMIT,
                torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path(__file__), ROOT/'moonep/sampled_runtime.py', ROOT/'moonep/csrc/sampled_runtime.cu')},
                variants=[dict(name=n, grid=g) for n, g in VARIANTS],
                workloads=[dict(name=n, words=w, op=o, calls=c) for n,w,o,c in WORKLOADS],
                grid_semantics='0: production four-launch; -1: legacy all-fused auto (diagnostic override); positive: forced cooperative grid, including grid=1. Production selective fusion uses 0 above 256 input words and -1 at or below 256.',
                order='For each sample, reverse workload and variant order on odd samples; keep every raw sample.',
                timing='External CUDA events inside Graph, after two untimed COMPLETE alignment collectives. One event pair per complete collective; max of rank medians only in summary.',
                input='Every Graph replay uses a new phase and distinct rank/round/element values. All calls, including warm alignment, restore their input; SUM/MAX operate in place.',
                excluded_from_call_timing=['input-reset copy', 'two alignment collectives', 'host launch', 'output validation', 'object gathers', 'allocation', 'JIT'],
                included_in_call_timing=['payload publication', 'first peer barrier', 'peer reads/reduction', 'read-completion peer barrier', 'cooperative grid barriers when selected'],
                group_timing='A separate event interval includes input reset plus all measured calls; alignment remains outside.',
                limitations=['Warm aligned communication microbenchmark, not full-planner or end-to-end training time.',
                             'Grid size limits launched CTAs; it does not pin physical SM IDs or measure occupancy.']))
            save()
        ctx = dict(rank=rank, group=None, meta_buf=torch.empty(1, device='cuda', dtype=torch.int32))
        runtime = BoundedPlannerRuntime(ctx, sm_limit=LIMIT, fuse=False, max_words=max(w for _,w,_,_ in WORKLOADS))
        work = {spec[0]: Workload(spec) for spec in WORKLOADS}
        # All ranks use the same compilation/capture sequence. Only the private
        # diagnostic entry selects a different grid; runtime.fuse stays False.
        dist.barrier()
        for name, _, _, _ in WORKLOADS:
            for variant, grid in VARIANTS:
                captures[name, variant] = capture(runtime, work[name], grid)
        phase = 0
        for sample in range(args.samples):
            specs = WORKLOADS if sample % 2 == 0 else tuple(reversed(WORKLOADS))
            variants = VARIANTS if sample % 2 == 0 else tuple(reversed(VARIANTS))
            for name, words, op, calls in specs:
                w = work[name]
                for position, (variant, grid) in enumerate(variants):
                    phase += 1
                    value = inputs(phase, rank, calls, words)
                    w.seed.copy_(torch.from_numpy(value))
                    capture_info = captures[name, variant]
                    capture_info['graph'].replay()
                    capture_info['group_end'].synchronize()
                    call_us = [begin.elapsed_time(end) * 1000
                               for begin, end in zip(capture_info['starts'], capture_info['ends'])]
                    group_us = capture_info['group_start'].elapsed_time(capture_info['group_end']) * 1000
                    observed = w.output.cpu().numpy().copy()
                    oracle = expected(phase, calls, words, op)
                    local_result = dict(rank=rank, phase=phase, passed=bool(np.array_equal(observed, oracle)),
                        call_us=call_us, total_collective_us=sum(call_us), group_with_input_reset_us=group_us,
                        input_sha256=digest(value), output_sha256=digest(observed), expected_sha256=digest(oracle))
                    all_results = gather(local_result, rank)
                    ok = [True]
                    if rank == 0:
                        row = dict(sample=sample, phase=phase, workload=name, variant=variant, grid=grid,
                                   position_in_variant_order=position, per_rank=all_results)
                        record['samples'].append(row)
                        ok[0] = ([x['rank'] for x in all_results] == list(range(R)) and
                                 all(x['phase'] == phase and x['passed'] and
                                     all(math.isfinite(y) and y > 0 for y in x['call_us']) and
                                     math.isfinite(x['group_with_input_reset_us']) and
                                     x['group_with_input_reset_us'] + 1e-3 >= x['total_collective_us']
                                     for x in all_results) and
                                 len({x['input_sha256'] for x in all_results}) == R)
                        save()
                    dist.broadcast_object_list(ok, src=0)
                    assert ok[0], f'incorrect peer collective: {name}/{variant}, phase {phase}'
            if rank == 0:
                print(json.dumps(dict(sample=sample, phases_completed=phase, passed=True)), flush=True)
        if rank == 0:
            record['summary'] = summarize(record['samples'], args.samples)
            record['status'] = 'complete'; save()
    except Exception:
        failure = dict(rank=rank, traceback=traceback.format_exc())
        args.output.with_name(args.output.stem + f'_rank{rank}_failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        if rank == 0 and record is not None:
            record['status']='failed'; record['failures'].append(failure); save()
        raise
    finally:
        if captures:
            torch.cuda.synchronize()
            for value in captures.values():
                value['graph'].reset()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
