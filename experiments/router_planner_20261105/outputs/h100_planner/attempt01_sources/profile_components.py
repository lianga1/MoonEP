"""Diagnostic eager Torch Profiler traces, never accepted latency measurements.

Run only after the unprofiled benchmark and with exclusive GPU access:
  python experiments/h100_planner/profile_components.py --output outputs/profile_layer0

Default scope is the exchange component; subset mining and lookup construction
are prepared before capture. Use --scope prepare_exchange to include them, or
--scope seed_exchange_output for the common seed + exchange + output pipeline.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from aggregate import AggregateRunner
from bundle_lane8 import Bundle8LanePlanner
from subset_cuda import SubsetRunner


def summarize_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    if not kernels:
        raise RuntimeError("No CUDA kernel events: profiler trace is not usable")
    grouped = defaultdict(lambda: dict(count=0, total_us=0.0))
    for event in kernels:
        row = grouped[event["name"]]
        row["count"] += 1
        row["total_us"] += float(event["dur"])
    intervals = sorted((float(e["ts"]), float(e["ts"] + e["dur"])) for e in kernels)
    begin, end = intervals[0]
    union = 0.0
    for left, right in intervals[1:]:
        if left > end:
            union += end - begin
            begin, end = left, right
        else:
            end = max(end, right)
    union += end - begin
    rows = [dict(name=name, **row, mean_us=row["total_us"] / row["count"])
            for name, row in grouped.items()]
    rows.sort(key=lambda row: -row["total_us"])
    memory = [e for e in events if e.get("cat") in ("gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    return dict(kernel_events=len(kernels), kernel_duration_sum_us=sum(e["dur"] for e in kernels),
                kernel_busy_union_us=union,
                kernel_envelope_us=max(right for _, right in intervals) - intervals[0][0],
                gpu_memory_event_count=len(memory), gpu_memory_duration_sum_us=sum(e["dur"] for e in memory),
                kernels=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--variants", default="bundle8,subset_parallel,subset_indexed,aggregate_soa")
    parser.add_argument("--scope", choices=("exchange", "prepare_exchange", "seed_exchange_output"), default="exchange")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Choose a fresh output directory to retain the previous diagnostic run")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    x = np.load(args.root / f"outputs/trace_audit/layer_{args.layer:02d}.npz")["topk"].astype(np.int32)
    support_file = np.load(args.root / f"outputs/replay/layer{args.layer:02d}_b8_small256_global_4.npz")
    seed = np.load(args.root / f"outputs/rt2_b8_comparison/layer{args.layer:02d}_native_rt0.npz")["owner"].astype(np.int32)
    xt, ot, st = (torch.as_tensor(value, device="cuda") for value in (x, seed, support_file["support"]))
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__, cuda=torch.version.cuda, layer=args.layer, shape=list(x.shape),
                    scope=args.scope, warmup=args.warmup, repeats=args.repeats, physical_gpus=1, logical_ranks=8,
                    warning="Profiled eager diagnostics only. Kernel sum, busy union and envelope are distinct; none is a replacement for unprofiled benchmark latency.",
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob("*") if p.is_file()})
    report = dict(metadata=metadata, rows=[], failures=[])
    baseline = materializer = None
    if args.scope == "seed_exchange_output":
        from baseline_virtual import VirtualMoonEPPlanner
        from materialize_cuda import CudaMaterializer
        baseline, materializer = VirtualMoonEPPlanner(xt), CudaMaterializer(xt, st)

    for name in args.variants.split(","):
        try:
            if name == "bundle8":
                runner = Bundle8LanePlanner(xt, st, x.shape[1])
                expected = support_file["owner"]
                exchange = runner.run
            elif name == "aggregate_soa":
                runner = AggregateRunner(xt, st, coalesced=True)
                expected = np.load(args.root / f"outputs/aggregate_exchange_multipair/layer{args.layer:02d}_mixed_sizes.npz")["owner"]
                exchange = runner.run
            elif name in ("subset_parallel", "subset_indexed"):
                runner = SubsetRunner(xt, st, h=2, g=8)
                expected = np.load(args.root / f"outputs/h100_subset_reference/layer{args.layer:02d}.npz")["owner"]
                if name == "subset_indexed":
                    runner.prepare_indexed(ot)
                    exchange = runner.run_indexed if args.scope == "exchange" else runner.full_indexed
                else:
                    runner.prepare(ot)
                    exchange = runner.run_parallel if args.scope == "exchange" else runner.full_parallel
            else:
                raise ValueError(name)
            state = {}

            def execute():
                initial = ot if baseline is None else baseline.run_seed().reshape(x.shape)
                state["owner"] = exchange(initial)
                if materializer is not None:
                    state["materialized"] = materializer.run(state["owner"])

            for _ in range(args.warmup):
                execute()
            torch.cuda.synchronize()
            assert np.array_equal(state["owner"].cpu().numpy().reshape(x.shape), expected)
            trace = args.output / f"layer{args.layer:02d}_{name}_{args.scope}.trace.json"
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, profile_memory=False, with_stack=False) as profiler:
                for step in range(args.repeats):
                    with torch.profiler.record_function(f"{name}/{args.scope}/step{step}"):
                        execute()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(trace))
            assert np.array_equal(state["owner"].cpu().numpy().reshape(x.shape), expected)
            row = dict(variant=name, exact_cpu=True, trace=str(trace), **summarize_trace(trace))
            report["rows"].append(row)
            print(json.dumps(dict(variant=name, kernel_events=row["kernel_events"], top_kernels=row["kernels"][:10])), flush=True)
        except Exception as exc:
            report["failures"].append(dict(variant=name, error=repr(exc), traceback=traceback.format_exc()))
            print("PROFILE_FAILED", name, repr(exc), flush=True)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
