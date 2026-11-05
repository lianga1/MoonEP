#!/usr/bin/env python3
"""Offline, conservative attribution of sampled-planner Torch Chrome traces.

python3 experiments/planner_stage_profile/analyze_traces.py \
  --trace-dir /path/to/traces --output /path/to/fresh_analysis.json

Standard library only. Inputs remain unchanged. Output creation is exclusive.
GPU duration sum, interval union and first-to-last span are different metrics;
none is the unprofiled planner wall latency. Eager time is never transferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


FILE_RE = re.compile(r"layer(\d+)_rank(\d+)_(eager_mapping|graph_3replays)\.json(?:\.gz)?$")
STAGE_RE = re.compile(r"^planner_stage/(\d+)/(.+)$")
REPLAY_RE = re.compile(r"^uninstrumented_graph_replay/(\d+)$")
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATEGORIES = {"cpu_op", "user_annotation", "python_function"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(event, field):
    # IDs are local to ONE capture. Never join numeric IDs across trace files.
    value = event.get("args", {}).get(field)
    if value in (None, "", 0, "0"):
        return None
    return str(value)


def end(event):
    return float(event["ts"]) + float(event.get("dur", 0))


def signature(event):
    args = event.get("args", {})
    # Captured stream IDs and correlation IDs change; compare actual operation,
    # launch shape, shared memory and memcpy size when supplied in BOTH files.
    return json.dumps(dict(category=event.get("cat"), name=event.get("name"),
        grid=args.get("grid"), block=args.get("block"), shared=args.get("shared memory"),
        bytes=args.get("bytes")), sort_keys=True)


def kind(stage):
    if stage == "pre_sync" or stage.startswith("publication_barrier"):
        return "sync"
    if stage in ("sample_sum", "status_max") or stage.endswith("allgather"):
        return "collective"
    if stage in ("input_copy", "status_merge", "fallback_select", "publication_clear"):
        return "glue"
    return "kernel"


def union_us(events):
    intervals = sorted((float(e["ts"]), end(e)) for e in events if float(e.get("dur", 0)) > 0)
    if not intervals:
        return 0.0
    total, left, right = 0.0, *intervals[0]
    for begin, finish in intervals[1:]:
        if begin > right:
            total += right-left
            left, right = begin, finish
        else:
            right = max(right, finish)
    return total + right-left


def metrics(events, divisor=None):
    duration = sum(float(e.get("dur", 0)) for e in events)
    busy = union_us(events)
    span = max(map(end, events))-min(float(e["ts"]) for e in events) if events else 0.0
    result = dict(calls=len(events), sum_us=duration, union_us=busy, span_us=span,
                  overlap_us=max(0.0, duration-busy), internal_gap_us=max(0.0, span-busy))
    if divisor:
        result.update(calls_per_replay=len(events)/divisor, sum_us_per_replay=duration/divisor,
                      union_us_per_replay=busy/divisor,
                      amortized_span_us_per_replay=span/divisor)
    return result


class Trace:
    def __init__(self, path):
        self.path = path
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as file:
            self.document = json.load(file)
        raw = self.document.get("traceEvents", []) if isinstance(self.document, dict) else self.document
        self.events = []
        for index, event in enumerate(raw):
            if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
                continue
            item = dict(event, event_index=index)
            if float(item["dur"]) < 0:
                raise ValueError(f"Negative duration at event {index} in {path}")
            self.events.append(item)
        self.gpu = sorted((e for e in self.events if e.get("cat") in GPU_CATEGORIES),
                          key=lambda e: (float(e["ts"]), e["event_index"]))
        self.by_external = defaultdict(list)
        self.by_correlation = defaultdict(list)
        for event in self.events:
            category = event.get("cat", "")
            external, correlation = identifier(event, "External id"), identifier(event, "correlation")
            if external and category in CPU_CATEGORIES:
                self.by_external[external].append(event)
            if correlation and category in ("cuda_runtime", "cuda_driver"):
                self.by_correlation[correlation].append(event)
        self.stages = [e for e in self.events if e.get("cat") == "user_annotation"
                       and STAGE_RE.match(e.get("name", ""))]
        self.replays = [e for e in self.events if e.get("cat") == "user_annotation"
                        and REPLAY_RE.match(e.get("name", ""))]

    def provenance(self):
        metadata = self.document if isinstance(self.document, dict) else {}
        return dict(path=str(self.path), sha256=sha(self.path),
                    trace_id=metadata.get("trace_id"), deviceProperties=metadata.get("deviceProperties"),
                    distributedInfo=metadata.get("distributedInfo"),
                    cuda_driver_version=metadata.get("cuda_driver_version"),
                    cuda_runtime_version=metadata.get("cuda_runtime_version"),
                    cupti_version=metadata.get("cupti_version"), gpu_event_count=len(self.gpu))

    def linked_annotation(self, gpu, annotations):
        """GPU -> same-trace correlation/external ID -> containing CPU label."""
        references = {}
        for event in self.by_external.get(identifier(gpu, "External id"), []):
            references[event["event_index"]] = (event, "gpu_external_id")
        for launch in self.by_correlation.get(identifier(gpu, "correlation"), []):
            references[launch["event_index"]] = (launch, "gpu_correlation_id")
            for cpu in self.by_external.get(identifier(launch, "External id"), []):
                references[cpu["event_index"]] = (cpu, "launch_external_id")
        candidates, evidence = set(), []
        for cpu, method in references.values():
            for annotation in annotations:
                same_thread = cpu.get("pid") == annotation.get("pid") and cpu.get("tid") == annotation.get("tid")
                contained = (float(annotation["ts"]) <= float(cpu["ts"])
                             and end(cpu) <= end(annotation)+0.001)
                if cpu["event_index"] == annotation["event_index"] or (same_thread and contained):
                    candidates.add(annotation["name"])
                    evidence.append(dict(method=method, cpu_event_index=cpu["event_index"],
                        cpu_name=cpu.get("name"), annotation_event_index=annotation["event_index"],
                        annotation=annotation["name"], call_stack=cpu.get("args", {}).get("Call stack")))
        return sorted(candidates), evidence


def event_record(event, stage=None, replay=None, method=None, reason=None, evidence=None):
    args = event.get("args", {})
    return dict(event_index=event["event_index"], name=event["name"], category=event.get("cat"),
        ts_us=float(event["ts"]), duration_us=float(event["dur"]),
        device=args.get("device", event.get("pid")), stream=args.get("stream", event.get("tid")),
        correlation=identifier(event, "correlation"), external_id=identifier(event, "External id"),
        grid=args.get("grid"), block=args.get("block"), stage=stage, replay=replay,
        attribution_method=method, unmapped_reason=reason, source=evidence or [])


def analyze_pair(mapping_path, graph_path, expected_replays):
    eager, graph = Trace(mapping_path), Trace(graph_path)
    warnings = []
    if len(eager.stages) != 36:
        warnings.append(f"Expected 36 eager stage labels, found {len(eager.stages)}")
    if not eager.gpu or not graph.gpu:
        raise ValueError("Both traces must contain nonzero GPU activity")
    eager_rows, by_signature = [], defaultdict(list)
    for event in eager.gpu:
        labels, evidence = eager.linked_annotation(event, eager.stages)
        stage = STAGE_RE.match(labels[0]).group(2) if len(labels) == 1 else None
        row = event_record(event, stage=stage,
            method="same_trace_external_or_correlation_to_cpu_stage" if stage else None,
            reason=None if stage else ("ambiguous_stage_labels" if labels else "no_linked_cpu_stage"),
            evidence=evidence)
        eager_rows.append(row)
        by_signature[signature(event)].append((stage, row))
    replay_ids = [int(REPLAY_RE.match(e["name"]).group(1)) for e in graph.replays]
    valid_replays = len(replay_ids) == expected_replays and sorted(replay_ids) == list(range(expected_replays))
    if not valid_replays:
        warnings.append(f"Expected unique replay labels 0..{expected_replays-1}, observed {replay_ids}; normalization disabled")
    divisor = expected_replays if valid_replays else None
    graph_rows, grouped = [], defaultdict(list)
    for event in graph.gpu:
        labels, evidence = graph.linked_annotation(event, graph.replays)
        replay = int(REPLAY_RE.match(labels[0]).group(1)) if len(labels) == 1 else None
        row = event_record(event, replay=replay, evidence=evidence)
        graph_rows.append(row)
        if replay is not None:
            grouped[replay].append((event, row))
    if sorted(grouped) != list(range(expected_replays)):
        valid_replays = False
        divisor = None
        warnings.append(f"Not every expected replay has ID-linked GPU activity: {sorted(grouped)}; normalization disabled")
    sequence_checks = []
    eager_signatures = [signature(e) for e in eager.gpu]
    stage_indices = {STAGE_RE.match(e["name"]).group(2): int(STAGE_RE.match(e["name"]).group(1))
                     for e in eager.stages}
    observed_stage_order = [stage_indices.get(row["stage"]) for row in eager_rows]
    order_known = all(index is not None for index in observed_stage_order)
    stage_order_monotonic = order_known and observed_stage_order == sorted(observed_stage_order)
    cross_stage_overlap = False
    previous_stage, previous_end = None, float("-inf")
    for event, row in zip(eager.gpu, eager_rows):
        if row["stage"] != previous_stage and float(event["ts"]) < previous_end-0.001:
            cross_stage_overlap = True
        previous_stage = row["stage"]
        previous_end = max(previous_end, end(event))
    sequence_transfer_safe = stage_order_monotonic and not cross_stage_overlap
    for replay, pairs in sorted(grouped.items()):
        # Full exact ordered-signature equality is stricter than name matching.
        # Eager IDs establish stages; graph IDs establish replay boundaries.
        # Transferring by ordinal is explicitly reported as cross-trace evidence,
        # not as a nonexistent shared external-ID relation across captures.
        exact = [signature(e) for e, _ in pairs] == eager_signatures
        sequence_checks.append(dict(replay=replay, whole_sequence_exact=exact,
                                    stage_transfer_allowed=exact and sequence_transfer_safe,
                                    eager_stage_order_monotonic=stage_order_monotonic,
                                    eager_cross_stage_overlap=cross_stage_overlap,
                                    eager_events=len(eager_signatures), graph_events=len(pairs)))
        for index, (event, row) in enumerate(pairs):
            labels, direct = graph.linked_annotation(event, graph.stages)
            if len(labels) == 1:
                row.update(stage=STAGE_RE.match(labels[0]).group(2),
                           attribution_method="same_graph_trace_ids", source=row["source"]+direct)
            elif len(labels) > 1:
                row["unmapped_reason"] = "ambiguous_direct_graph_stage_labels"
            elif exact and sequence_transfer_safe and eager_rows[index]["stage"] is not None:
                mapping = eager_rows[index]
                row.update(stage=mapping["stage"], attribution_method="exact_whole_replay_signature_sequence",
                    source=row["source"]+[dict(mapping_trace=str(mapping_path),
                        mapping_event_index=mapping["event_index"], mapping_stage=mapping["stage"],
                        ordinal=index, eager_source=mapping["source"])])
            else:
                matches = by_signature.get(signature(event), [])
                candidates = {stage for stage, _ in matches}
                if len(candidates) == 1 and None not in candidates:
                    stage = next(iter(candidates))
                    row.update(stage=stage, attribution_method="unique_eager_launch_signature",
                        source=row["source"]+[dict(mapping_trace=str(mapping_path),
                            mapping_event_indices=[item["event_index"] for _, item in matches],
                            mapping_stage=stage)])
                else:
                    row["unmapped_reason"] = "ambiguous_eager_signature" if matches else "no_eager_signature"
                    row["candidate_stages"] = sorted(str(stage) for stage in candidates)
    for row in graph_rows:
        if row["replay"] is None:
            row["unmapped_reason"] = "no_unique_graph_replay_via_ids"
        elif row["stage"] is None and row["unmapped_reason"] is None:
            row["unmapped_reason"] = "unknown"

    def grouped_metrics(events, rows, denominator):
        by_stage, by_name, nccl, barriers = defaultdict(list), defaultdict(list), [], []
        for annotation in eager.stages:
            by_stage[STAGE_RE.match(annotation["name"]).group(2)]
        for event, row in zip(events, rows):
            stage = row["stage"] or "UNMAPPED"
            by_stage[stage].append(event)
            by_name[(stage, event["name"], event.get("cat"), row["attribution_method"])].append((event, row))
            if "nccl" in event["name"].lower():
                nccl.append(event)
            if row["stage"] is not None and kind(row["stage"]) == "sync":
                barriers.append(event)
        stages = [dict(stage=stage, kind=kind(stage) if stage != "UNMAPPED" else "unknown",
                       gpu_activity=metrics(items, denominator),
                       kernels=metrics([e for e in items if e.get("cat") == "kernel"], denominator))
                  for stage, items in sorted(by_stage.items())]
        kernels = [dict(stage=key[0], name=key[1], category=key[2], attribution_method=key[3],
                        **metrics([e for e, _ in items], denominator),
                        source_event_indices=[r["event_index"] for _, r in items])
                   for key, items in by_name.items()]
        kernels.sort(key=lambda row: row["sum_us"], reverse=True)
        return dict(stages=stages, kernel_and_activity_table=kernels,
                    nccl_name_matched=metrics(nccl, denominator),
                    explicit_barrier_stage_gpu=metrics(barriers, denominator))

    graph_metrics = grouped_metrics(graph.gpu, graph_rows, divisor)
    per_replay = [dict(replay=replay, kernel_metrics=metrics([e for e, _ in pairs if e.get("cat") == "kernel"]),
                       gpu_activity_metrics=metrics([e for e, _ in pairs])) for replay, pairs in sorted(grouped.items())]
    graph_metrics.update(kernel_metrics=metrics([e for e in graph.gpu if e.get("cat") == "kernel"], divisor),
        gpu_activity_metrics=metrics(graph.gpu, divisor), per_replay=per_replay,
        mean_replay_kernel_span_us=statistics.mean(row["kernel_metrics"]["span_us"] for row in per_replay)
                                  if valid_replays and len(per_replay) == expected_replays else None,
        mapping_coverage=dict(total=len(graph_rows), mapped=sum(r["stage"] is not None for r in graph_rows),
            unmapped=sum(r["stage"] is None for r in graph_rows),
            methods=dict(Counter(r["attribution_method"] or "UNMAPPED" for r in graph_rows))),
        sequence_checks=sequence_checks, replay_labels=replay_ids, normalization_divisor=divisor,
        raw_event_attribution=graph_rows)
    return dict(mapping_source=eager.provenance(), graph_source=graph.provenance(), warnings=warnings,
        stage_labels=[dict(index=int(STAGE_RE.match(e["name"]).group(1)),
                           stage=STAGE_RE.match(e["name"]).group(2), event_index=e["event_index"])
                      for e in eager.stages],
        eager_diagnostic=dict(**grouped_metrics(eager.gpu, eager_rows, None),
                              raw_event_attribution=eager_rows), graph=graph_metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", help="Comma-separated layer IDs; default discovers layers")
    parser.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--expected-replays", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.expected_replays <= 0:
        raise ValueError("expected-replays must be positive")
    files = defaultdict(list)
    for path in args.trace_dir.rglob("*.json*"):
        match = FILE_RE.fullmatch(path.name)
        if match:
            files[(int(match.group(1)), int(match.group(2)), match.group(3))].append(path)
    layers = list(map(int, args.layers.split(","))) if args.layers else sorted({key[0] for key in files})
    ranks = list(map(int, args.ranks.split(",")))
    if not layers:
        raise ValueError("No matching stage-profile trace files found")
    result = dict(status="complete", unit="microseconds (Chrome trace ts/dur)",
        analyzer_sha256=sha(Path(__file__)), pairs=[], failures=[],
        metric_contract={
            "sum_us": "Sum of individual GPU durations; double-counts overlap, NOT wall latency.",
            "union_us": "Union of GPU activity intervals across all streams of this rank capture.",
            "span_us": "Earliest GPU start to latest GPU end, including internal gaps; excludes work outside captured GPU activity.",
            "per_replay": "Sum and union divided only when all expected replay labels are present. Amortized span also includes gaps between replays.",
            "mean_replay_kernel_span_us": "Mean of each correlation-identified replay's first-to-last kernel span; different from total capture span divided by replays.",
            "attribution": "IDs are joined only inside a trace. Graph replay stage transfer requires full ordered signature equality or a uniquely attributed eager signature; otherwise UNMAPPED.",
            "limits": "Profiled durations are diagnostics, not uninstrumented planner latency. Different rank clocks are never merged. Explicit barriers and NCCL kernels include waits; native_seed also embeds NVLink synchronization not separately identifiable from its kernel duration.",
        })
    for layer in layers:
        for rank in ranks:
            try:
                paths = [files[(layer, rank, category)] for category in ("eager_mapping", "graph_3replays")]
                if any(len(items) != 1 for items in paths):
                    raise ValueError(f"Exactly one mapping and graph file required, found {[[str(p) for p in items] for items in paths]}")
                pair = analyze_pair(paths[0][0], paths[1][0], args.expected_replays)
                result["pairs"].append(dict(layer=layer, rank=rank, **pair))
            except Exception as error:
                result["failures"].append(dict(layer=layer, rank=rank, error=repr(error)))
    if result["failures"]:
        result["status"] = "partial"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output),
                          pairs=len(result["pairs"]), failures=len(result["failures"]))))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
