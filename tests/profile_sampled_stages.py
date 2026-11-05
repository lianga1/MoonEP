"""Eight-physical-rank sampled planner timing and stage diagnostics.

Run from the source checkout with:
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m tests.profile_sampled_stages --fixture-root /path/to/kernel_inputs \
    --output /path/to/fresh_attempt.json --profile-dir /path/to/fresh_traces

The unchanged benchmark helper measures native and sampled full paths first.
Stage events and profiler traces are DIAGNOSTICS: their instrumentation changes
execution. They must not replace the uninstrumented complete-path latency.
Only local runtime routes enter planning; saved fixture plans are never used.
All output paths must be fresh. Partial records and per-rank failures survive.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import time
import traceback
import uuid

import numpy as np
import torch
import torch.distributed as dist

from moonep import Buffer
from moonep.inter_rank_sync import launch_inter_rank_sync
from moonep.planning import allocate_planning_outputs, launch_moon_planning, launch_planning
from tests.benchmark_sampled_group import (
    E, FIELDS, K, R, S, ROOT, audit_plans, gather_root, measure,
    plan_snapshot, precompile_planner, read_local_routes, sha,
)


def source_manifest():
    files = [Path(__file__), ROOT / "tests" / "benchmark_sampled_group.py"]
    files += [p for p in (ROOT / "moonep").rglob("*")
              if p.is_file() and p.suffix in (".py", ".cu", ".cuh")]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}


def cpu_tensor(tensor):
    return tensor.detach().cpu().numpy().copy()


def complete_snapshot(plan, cu, ctx, adapter=None):
    """Untimed, valid output/state only; unused history capacity is undefined."""
    snapshot = plan_snapshot(plan, cu, ctx, adapter=adapter)
    if adapter is None:
        return snapshot
    residency, mat = adapter.residency, adapter.materializer
    history_count = int(residency.history_count.item())
    package_count = int(residency.package_count.item())
    if not 0 <= history_count <= residency.history.shape[0]:
        raise AssertionError("invalid history_count")
    if not 0 <= package_count <= residency.packages.shape[0]:
        raise AssertionError("invalid package_count")
    start = int(ctx["rank"]) * int(ctx["meta_chunk_padded"]) + int(ctx["SRC_INFO_OFF"])
    snapshot.update(
        routes=cpu_tensor(adapter.routes),
        seed_owner=cpu_tensor(adapter.seed_owner),
        residency_status=int(residency.status.item()),
        materializer_status=int(mat.status.item()),
        history_count=history_count,
        history=cpu_tensor(residency.history[:history_count]),
        package_count=package_count,
        packages=cpu_tensor(residency.packages[:package_count]),
        sample_packet=cpu_tensor(residency.packet),
        samples=cpu_tensor(residency.samples),
        selected_copies=cpu_tensor(residency.experts_to_copy),
        exchange_stats=cpu_tensor(adapter.exchange.stats),
        source_info=cpu_tensor(ctx["meta_buf"][start:start + int(ctx["NvS"])]),
        candidate={name: cpu_tensor(getattr(mat.plan, name)) for name in FIELDS},
        candidate_cu=cpu_tensor(mat.cu),
    )
    return snapshot


def compare_snapshots(expected, actual, prefix=""):
    """Return all mismatching names, not only the first differing tensor."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return [prefix + "/keys"]
        return [name for key in expected
                for name in compare_snapshots(expected[key], actual[key], prefix + "/" + key)]
    if isinstance(expected, np.ndarray):
        ok = (isinstance(actual, np.ndarray) and expected.dtype == actual.dtype
              and expected.shape == actual.shape and np.array_equal(expected, actual))
    else:
        ok = expected == actual
    return [] if ok else [prefix]


def output_hashes(snapshot):
    """Store small hashes of every compared output rather than all route data."""
    def digest(value):
        if isinstance(value, dict):
            return {key: digest(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return dict(shape=list(value.shape), dtype=str(value.dtype),
                        sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return value
    return digest(snapshot)


class SplitPlanner:
    """Exact expansion of SampledGroupPlanner.run plus the benchmark pre-sync.

    The same adapter, buffers, extension calls and collective order are used.
    Python validation in the production wrappers is bypassed only where a
    wrapper needs separate timer boundaries; the untimed equality/audit checks
    cover the actual fixed-shape inputs. No algorithm or allocation is changed.
    """
    def __init__(self, ctx, adapter, topk, tpe, cu, plan):
        self.stages = []
        self.events = None
        self.ctx, self.adapter = ctx, adapter
        a, res, ex, mat = adapter, adapter.residency, adapter.exchange, adapter.materializer

        def add(name, kind, fn, note=""):
            self.stages.append(dict(name=name, kind=kind, fn=fn, note=note))

        add("pre_sync", "sync", lambda: launch_inter_rank_sync(ctx),
            "Native peer-mapped NVLink rank synchronization included in the baseline timer.")
        add("input_copy", "glue", lambda: a.routes.copy_(topk.view_as(a.routes)))
        add("native_seed", "kernel",
            lambda: launch_moon_planning(ctx, topk, tpe, a.seed_cu, a.seed),
            "Native planner includes internal NVLink synchronization; this is not pure local compute.")
        add("decode_owner", "kernel", lambda: mat.decode_owner(a.seed.dst, a.seed_owner))
        add("residency_gather_local", "kernel",
            lambda: res.gather_local(a.routes, a.seed_owner, a.seed.experts_to_copy))
        add("sample_sum", "collective",
            lambda: dist.all_reduce(res.packet, op=dist.ReduceOp.SUM, group=res.group, async_op=False),
            "Live sampled-route/owner packet SUM across all eight physical ranks.")
        add("residency_select", "kernel", lambda: res.select(a.seed.experts_to_copy),
            "Multiple CUDA kernels; use eager mapping trace for internal kernel attribution.")
        add("materialize_prepare_support", "kernel", lambda: mat.prepare_support(a.support))
        add("exchange_prepare", "kernel", ex.prepare_support)
        add("exchange_reset", "kernel", lambda: ex.reset(a.seed_owner))
        for step in range(a.rounds):
            add(f"round{step + 1}_propose", "kernel", lambda step=step: ex.propose(step))
            add(f"round{step + 1}_counts_allgather", "collective",
                lambda: dist.all_gather_into_tensor(a.gathered_counts.view(-1), ex.counts, group=a.group))
            add(f"round{step + 1}_quota", "kernel",
                lambda step=step: ex.quota(step, a.gathered_counts))
            add(f"round{step + 1}_apply", "kernel", ex.apply)
        add("materialize_prepare", "kernel", lambda: mat.ext.prepare(
            a.routes, a.owner, a.seed_owner, mat.native_order, mat.lut,
            mat.keys_in, mat.keys_out, mat.ids_in, mat.ids_out,
            mat.starts, mat.counts, mat.workspace, mat.duplicates, mat.status,
            mat.R, mat.E, mat.K, mat.reuse_native_order, mat.sort_bits,
        ), "Includes stable order preparation / CUB sorting and local expert counts.")
        add("materialize_counts_allgather", "collective",
            lambda: dist.all_gather_into_tensor(mat.gathered_counts.view(-1), mat.counts.view(-1), group=mat.group))
        add("materialize_layout_encode", "kernel", lambda: mat.ext.layout_encode(
            a.routes, a.owner, mat.ids_out, mat.starts, mat.gathered_counts,
            mat.lut, mat.copies, mat.offsets, mat.prefix, mat.totals, mat.duplicates,
            mat.plan.dst, mat.cu, mat.plan.zero_fill_ranges, mat.plan.remote_stats,
            mat.status, mat.R, mat.E, mat.B, mat.K, mat.rank, mat.padding, mat.NvS,
        ))

        def merge_status():
            torch.maximum(res.status, mat.status, out=a.status)
            torch.maximum(a.status, a.force_reject, out=a.status)

        def fallback():
            torch.eq(a.status, 0, out=a.valid)
            torch.where(a.valid, mat.cu, a.seed_cu, out=cu)
            for name in FIELDS:
                torch.where(a.valid, getattr(mat.plan, name), getattr(a.seed, name),
                            out=getattr(plan, name))

        add("status_merge", "glue", merge_status)
        add("status_max", "collective",
            lambda: dist.all_reduce(a.status, op=dist.ReduceOp.MAX, group=a.group))
        add("fallback_select", "glue", fallback,
            "GPU eq plus cu/FIELDS where selection; runs even when no fallback is needed.")
        stride, offset = int(ctx["meta_chunk_padded"]), int(ctx["SRC_INFO_OFF"])
        begin = mat.rank * stride + offset
        add("publication_clear", "glue", lambda: ctx["meta_buf"][begin:begin + mat.NvS].fill_(-1))
        add("publication_barrier_before", "sync", lambda: launch_inter_rank_sync(ctx))
        add("publication_write", "kernel",
            lambda: mat.ext.publish(plan.dst, ctx["meta_buf"], mat.rank, mat.NvS, stride, offset),
            "Writes provenance to peer-mapped NVLink metadata; not purely local compute.")
        add("publication_barrier_after", "sync", lambda: launch_inter_rank_sync(ctx))

    def manifest(self):
        return [dict(index=i, name=s["name"], kind=s["kind"], note=s["note"])
                for i, s in enumerate(self.stages)]

    def warm_events(self):
        # external=True generates explicit event-record nodes during capture.
        # Non-external events may collapse into graph dependencies and cannot
        # serve as independent per-replay stage timing endpoints.
        self.events = [torch.cuda.Event(enable_timing=True, external=True)
                       for _ in range(len(self.stages) + 1)]
        for event in self.events:
            event.record()
        self.events[-1].synchronize()

    def run(self, *, events=False, labels=False):
        if events:
            if self.events is None:
                raise RuntimeError("Call warm_events before graph capture")
            self.events[0].record()
        for index, stage in enumerate(self.stages):
            if labels:
                with torch.profiler.record_function(f"planner_stage/{index:02d}/{stage['name']}"):
                    stage["fn"]()
            else:
                stage["fn"]()
            if events:
                self.events[index + 1].record()


def measure_stages(split, samples):
    """Single replay per sample; all raw stage times are retained per rank."""
    split.warm_events()
    for _ in range(3):
        split.run(events=True)
    torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    raw = []
    try:
        with torch.cuda.graph(graph):
            split.run(events=True)
        graph.replay(); torch.cuda.synchronize(); dist.barrier()
        for sample in range(samples):
            dist.barrier(); torch.cuda.synchronize()
            tick = time.perf_counter()
            graph.replay()
            split.events[-1].synchronize()
            host_ms = (time.perf_counter() - tick) * 1000
            times = {stage["name"]: split.events[i].elapsed_time(split.events[i + 1])
                     for i, stage in enumerate(split.stages)}
            local = dict(rank=dist.get_rank(), stage_ms=times,
                         total_cuda_ms=split.events[0].elapsed_time(split.events[-1]),
                         sum_stage_ms=sum(times.values()), host_ms=host_ms)
            gathered = gather_root(local)
            if dist.get_rank() == 0:
                raw.append(dict(sample=sample, replays=1, per_rank=gathered))
    finally:
        torch.cuda.synchronize()
        graph.reset()
    if dist.get_rank() != 0:
        return None
    medians = []
    for rank in range(R):
        medians.append(dict(rank=rank,
            stage_ms={stage["name"]: statistics.median(
                item["per_rank"][rank]["stage_ms"][stage["name"]] for item in raw)
                for stage in split.stages},
            total_cuda_ms=statistics.median(item["per_rank"][rank]["total_cuda_ms"] for item in raw)))
    return dict(raw=raw, per_rank_medians=medians,
        max_rank_median_total_cuda_ms=max(item["total_cuda_ms"] for item in medians),
        max_rank_median_stage_ms={stage["name"]: max(item["stage_ms"][stage["name"]] for item in medians)
                                  for stage in split.stages},
        scope="Instrumented CUDA Graph, one replay/sample, explicit external event at every boundary.",
        warning="Diagnostic event nodes perturb execution. Max-rank stage medians need not share a rank/sample; do not sum them as a measured critical path.")


def export_trace(prof, directory, label):
    path = directory / (label + ".json")
    compressed = directory / (label + ".json.gz")
    if path.exists() or compressed.exists():
        raise FileExistsError(f"Trace path already exists: {path}")
    prof.export_chrome_trace(str(path))
    with path.open("rb") as source, gzip.open(compressed, "xb", compresslevel=6) as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    result = dict(path=str(compressed), sha256=sha(compressed),
                  raw_json_bytes=path.stat().st_size, gzip_bytes=compressed.stat().st_size)
    path.unlink()  # Full raw trace remains, losslessly compressed.
    return result


def profiler_traces(split, uninstrumented, directory, layer):
    """Eager stage-to-kernel mapping plus an uninstrumented warmed Graph trace."""
    rank = dist.get_rank()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    for _ in range(3):
        split.run()
    torch.cuda.synchronize(); dist.barrier()
    with torch.profiler.profile(activities=activities, record_shapes=True,
                                with_stack=False, profile_memory=False) as prof:
        split.run(labels=True)
        torch.cuda.synchronize()
    eager = export_trace(prof, directory, f"layer{layer:02d}_rank{rank}_eager_mapping")
    dist.barrier()
    for _ in range(3):
        uninstrumented()
    torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            uninstrumented()
        graph.replay(); torch.cuda.synchronize(); dist.barrier()
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    with_stack=False, profile_memory=False) as prof:
            for replay in range(3):
                with torch.profiler.record_function(f"uninstrumented_graph_replay/{replay}"):
                    graph.replay()
            torch.cuda.synchronize()
        graph_trace = export_trace(prof, directory, f"layer{layer:02d}_rank{rank}_graph_3replays")
    finally:
        torch.cuda.synchronize()
        graph.reset()
    dist.barrier()
    return dict(rank=rank, eager_mapping=eager, graph_3replays=graph_trace,
                warning="Profiler traces are attribution diagnostics, not formal latency measurements.")


def broadcast_outcome(outcome):
    packet = [outcome if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(packet, src=0)
    if not packet[0]["passed"]:
        raise AssertionError(packet[0]["error"])


def collective_local_check(label, callback):
    """Coordinate CPU-side failures before entering another GPU collective."""
    try:
        value = callback()
        local = dict(rank=dist.get_rank(), passed=True)
    except Exception:
        value = None
        local = dict(rank=dist.get_rank(), passed=False, traceback=traceback.format_exc())
    statuses = [None] * R
    dist.all_gather_object(statuses, local)
    failed = [item for item in statuses if not item["passed"]]
    if failed:
        raise RuntimeError(f"{label}: {json.dumps(failed)}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,7")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--replays-per-sample", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",")]
    if not layers or len(set(layers)) != len(layers) or any(layer not in range(8) for layer in layers):
        raise ValueError("layers must be unique original layer indices 0..7")
    if min(args.samples, args.replays_per_sample) < 1:
        raise ValueError("samples and replays-per-sample must be positive")
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    buffer = None
    record = None
    reserved = False
    phase = "initialization"

    def save():
        if rank == 0 and reserved and record is not None:
            # This run owns the file after exclusive creation. Updating its
            # partial result does not overwrite an earlier attempt.
            args.output.write_text(json.dumps(record, indent=2) + "\n")

    try:
        torch.cuda.set_device(local); torch.set_num_threads(1)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=20))
        if dist.get_world_size() != R or int(os.environ["LOCAL_WORLD_SIZE"]) != R:
            raise ValueError("This benchmark requires eight physical local ranks")
        identities = [None] * R
        props = torch.cuda.get_device_properties(local)
        dist.all_gather_object(identities, dict(rank=rank, host=socket.gethostname(),
            local_rank=local, device=torch.cuda.current_device(),
            uuid=str(getattr(props, "uuid", "unavailable"))))
        if len({(item["host"], item["local_rank"]) for item in identities}) != R:
            raise ValueError("Ranks do not identify eight distinct local device ordinals")
        known_uuids = [item["uuid"] for item in identities if item["uuid"] != "unavailable"]
        if known_uuids and len(set(known_uuids)) != R:
            raise ValueError("Ranks must use eight distinct physical GPU UUIDs")
        outcome = None
        if rank == 0:
            try:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as file:
                    file.write('{"status":"initializing"}\n')
                reserved = True
                if args.profile_dir is not None:
                    args.profile_dir.mkdir(parents=True, exist_ok=False)
                outcome = dict(passed=True)
            except Exception:
                outcome = dict(passed=False, error=traceback.format_exc())
        broadcast_outcome(outcome)
        record = dict(status="running", metadata=dict(
            baseline_for_comparison="81a5a31", physical_ranks=R, original_trace_R=R,
            S=S, N=R*S, E=E, K=K, B=8, rounds=4, global_samples=256,
            groups_per_class=2, sample_seed=20261026, hidden_allocated_but_not_transferred=256,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
            nccl=torch.cuda.nccl.version(), devices=identities,
            uninstrumented_samples=args.samples, uninstrumented_replays_per_sample=args.replays_per_sample,
            diagnostic_samples=args.samples, diagnostic_replays_per_sample=1,
            diagnostic_events="external=True, enable_timing=True; N+1 boundary event nodes",
            aggregation="Per rank sample median, then maximum rank; layer means only over listed layers.",
            excluded=["fixture load", "local expert histogram", "allocation", "JIT", "audit",
                      "expert weight transport", "token dispatch/combine", "dedup builder in dispatch",
                      "gradient payload", "GEMM", "optimizer"],
            timing_warning="Only uninstrumented variants use the unchanged full-planner latency protocol. Stage/profiler records perturb execution and cannot replace it.",
            source_files=source_manifest()), layers=[], failures=[])
        save()
        phase = "buffer_and_jit"
        buffer = Buffer(S, 256, K, E, R, B=8, enable_pdl=False,
                        planning_backend="sampled_group", planning_rounds=4,
                        planning_sample_seed=20261026, explicitly_destroy=True)
        ctx = buffer._require_ctx()
        precompile_planner(ctx)
        adapter = ctx["sampled_group_planner"]
        for layer in layers:
            phase = f"layer{layer:02d}_input"
            path = args.fixture_root / f"layer_{layer:02d}.npz"
            local_routes = collective_local_check(phase, lambda: read_local_routes(path, rank))
            routes = torch.from_numpy(local_routes).to("cuda")
            topk = routes.flatten()
            tpe = torch.bincount(topk.long(), minlength=E).int()
            audit_routes = gather_root(local_routes)
            native_plan, native_cu = allocate_planning_outputs(ctx)
            plan, cu = allocate_planning_outputs(ctx)

            def native():
                launch_inter_rank_sync(ctx)
                launch_moon_planning(ctx, topk, tpe, native_cu, native_plan)

            def sampled():
                launch_inter_rank_sync(ctx)
                launch_planning(ctx, topk, tpe, cu, plan)

            split = SplitPlanner(ctx, adapter, topk, tpe, cu, plan)
            row = dict(layer=layer, fixture=dict(name=path.name, sha256=sha(path)),
                       stages=split.manifest(), uninstrumented={}, correctness={})
            record["layers"].append(row); save()
            phase = f"layer{layer:02d}_uninstrumented"
            for name, fn in (("moon", native), ("sampled_group", sampled)):
                result = measure(fn, args.samples, args.replays_per_sample)
                if rank == 0:
                    row["uninstrumented"][name] = result; save()
            phase = f"layer{layer:02d}_equivalence"
            native(); torch.cuda.synchronize()
            native_items = gather_root(complete_snapshot(native_plan, native_cu, ctx))
            sampled(); torch.cuda.synchronize()
            expected = collective_local_check("production snapshot",
                lambda: complete_snapshot(plan, cu, ctx, adapter))
            production_items = gather_root(expected)
            split.run(); torch.cuda.synchronize()
            actual = collective_local_check("split snapshot",
                lambda: complete_snapshot(plan, cu, ctx, adapter))
            comparisons = gather_root(dict(rank=rank, mismatches=compare_snapshots(expected, actual)))
            split_items = gather_root(actual)
            outcome = None
            if rank == 0:
                try:
                    assert all(not item["mismatches"] for item in comparisons), comparisons
                    all_routes = np.stack(audit_routes)
                    row["correctness"].update(
                        split_exact_by_rank=comparisons,
                        native_audit=audit_plans(native_items, all_routes, 8),
                        production_audit=audit_plans(production_items, all_routes, 8, optimized=True),
                        split_audit=audit_plans(split_items, all_routes, 8, optimized=True),
                        output_hashes_by_rank=[output_hashes(item) for item in production_items],
                        package_count_by_rank=[item["package_count"] for item in production_items],
                        history_count_by_rank=[item["history_count"] for item in production_items])
                    assert all(np.array_equal(item["routes"], local)
                               for item, local in zip(production_items, audit_routes))
                    assert all(all(np.array_equal(b[name], o["native_seed"][name]) for name in FIELDS)
                               and np.array_equal(b["cu"], o["native_seed_cu"])
                               for b, o in zip(native_items, production_items))
                    row["correctness"]["native_seed_equals_direct_native"] = True
                    outcome = dict(passed=True)
                except Exception:
                    outcome = dict(passed=False, error=traceback.format_exc())
                    row["correctness"]["failure"] = outcome
                save()
            broadcast_outcome(outcome)
            phase = f"layer{layer:02d}_external_events"
            result = measure_stages(split, args.samples)
            if rank == 0:
                row["instrumented"] = result
                reference = row["uninstrumented"]["sampled_group"]["max_rank_median_cuda_ms"]
                event_time = result["max_rank_median_total_cuda_ms"]
                row["instrumentation_comparison"] = dict(
                    uninstrumented_ms=reference, instrumented_ms=event_time,
                    observed_delta_ms=event_time-reference,
                    observed_ratio=event_time/reference,
                    warning="Different replay batching (configured vs one) plus event-node perturbation; this delta is not a calibrated event-overhead subtraction.")
                save()
            graph_actual = collective_local_check("instrumented graph snapshot",
                lambda: complete_snapshot(plan, cu, ctx, adapter))
            comparisons = gather_root(dict(rank=rank, mismatches=compare_snapshots(expected, graph_actual)))
            graph_items = gather_root(graph_actual)
            outcome = None
            if rank == 0:
                try:
                    assert all(not item["mismatches"] for item in comparisons), comparisons
                    row["correctness"]["instrumented_graph_exact_by_rank"] = comparisons
                    row["correctness"]["instrumented_graph_audit"] = audit_plans(graph_items, all_routes, 8, optimized=True)
                    outcome = dict(passed=True)
                except Exception:
                    outcome = dict(passed=False, error=traceback.format_exc())
                    row["correctness"]["graph_failure"] = outcome
                save()
            broadcast_outcome(outcome)
            if args.profile_dir is not None:
                phase = f"layer{layer:02d}_profiler"
                traced = profiler_traces(split, sampled, args.profile_dir, layer)
                trace_items = gather_root(traced)
                post_trace = collective_local_check("post-profiler snapshot",
                    lambda: complete_snapshot(plan, cu, ctx, adapter))
                comparisons = gather_root(dict(rank=rank, mismatches=compare_snapshots(expected, post_trace)))
                outcome = None
                if rank == 0:
                    row["traces"] = trace_items
                    row["correctness"]["post_profiler_exact_by_rank"] = comparisons
                    outcome = dict(passed=all(not item["mismatches"] for item in comparisons),
                                   error=str(comparisons))
                    save()
                broadcast_outcome(outcome)
            row["completed"] = True; save()
            dist.barrier()
        if rank == 0:
            record["summary"] = dict(
                layers=layers,
                native_mean_layer_us=1000*statistics.mean(
                    row["uninstrumented"]["moon"]["max_rank_median_cuda_ms"] for row in record["layers"]),
                sampled_mean_layer_us=1000*statistics.mean(
                    row["uninstrumented"]["sampled_group"]["max_rank_median_cuda_ms"] for row in record["layers"]),
                all_layers_completed=all(row.get("completed", False) for row in record["layers"]))
            record["status"] = "complete"; save()
    except Exception:
        failure = dict(rank=rank, phase=phase, traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure_path = args.output.with_name(args.output.stem + f"_rank{rank}_failure.json")
        if failure_path.exists():
            failure_path = failure_path.with_name(failure_path.stem + "_" + uuid.uuid4().hex + ".json")
        with failure_path.open("x") as file:
            json.dump(failure, file, indent=2)
        if rank == 0 and record is not None:
            record["status"] = "failed"; record["failures"].append(failure); save()
        raise
    finally:
        # Every helper resets its graph before control reaches destruction.
        if buffer is not None:
            torch.cuda.synchronize()
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
