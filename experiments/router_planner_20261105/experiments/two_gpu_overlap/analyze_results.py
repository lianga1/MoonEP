"""Conservative paired accounting for two-rank planner/communication samples.

Input contract (normalization from a harness should be explicit):
{
  "schema_version": 1,
  "metadata": {"physical_ranks": 2,
    "timed_scope": "planner_weights_dispatch_combine",
    "hot_weights_same_version_and_mapping": true},
  "failures": [],
  "layers": [{"layer": 0, "audit_pass": true,
    "samples": [{"round": 0, "rank_times_ms": {
      "nativecoldserial": [1.1,1.2], "nativecoldearly": [1.0,1.1],
      "nativehot": [0.8,0.9], "optcoldserial": [1.0,1.1],
      "optcoldearly": [0.9,1.0], "opthot": [0.7,0.8]}}]}]
}

Every raw failure is preserved. Eligibility describes measured samples only;
neither bootstrap intervals nor a fallback promise guarantee future speedup.
The file uses only the Python standard library and can run on a preparation host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as stats


def quantile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("no samples")
    index = q * (len(values)-1)
    lo, hi = int(math.floor(index)), int(math.ceil(index))
    return values[lo] + (values[hi]-values[lo]) * (index-lo)


def rank_max(value, ranks=2):
    if len(value) != ranks:
        raise ValueError(f"expected {ranks} rank samples, got {len(value)}")
    if any(not math.isfinite(x) or x < 0 for x in value):
        raise ValueError("nonfinite or negative duration")
    return max(value)


def paired_comparison(samples, baseline, candidate, *, guardband_ms=0.0,
                      bootstrap_repeats=2000, seed=20261105):
    """Pair round first, take each mode's slowest rank, then subtract.

    Positive delta means the candidate is faster. Missing/duplicate rounds are
    errors rather than silently reduced cohorts. Interval is a paired bootstrap
    of mean delta, not a prediction interval for the next invocation.
    """
    if not samples:
        raise ValueError("no paired samples")
    rounds = [x["round"] for x in samples]
    if len(rounds) != len(set(rounds)):
        raise ValueError("duplicate round IDs")
    base = [rank_max(x["rank_times_ms"][baseline]) for x in samples]
    opt = [rank_max(x["rank_times_ms"][candidate]) for x in samples]
    delta = [b-o for b,o in zip(base,opt)]
    rng = random.Random(seed)
    resampled = [stats.mean(rng.choices(delta,k=len(delta)))
                 for _ in range(bootstrap_repeats)]
    lower, upper = quantile(resampled,0.025), quantile(resampled,0.975)
    # A confidence interval from one sample has no useful coverage; retain result
    # for diagnostics but never certify such a cohort.
    enough = len(delta) >= 10
    eligible = enough and min(delta) > guardband_ms and lower > guardband_ms
    return dict(baseline=baseline,candidate=candidate,n=len(delta),rounds=rounds,
        baseline_rank_max_ms=base,candidate_rank_max_ms=opt,paired_delta_ms=delta,
        baseline_median_ms=stats.median(base),candidate_median_ms=stats.median(opt),
        mean_delta_ms=stats.mean(delta),median_delta_ms=stats.median(delta),
        mean_delta_bootstrap_95pct_ms=[lower,upper],min_observed_delta_ms=min(delta),
        observed_positive_pairs=sum(x>0 for x in delta),
        minimum_sample_count_met=enough,
        guardband_ms=guardband_ms,measured_positive_gate_pass=eligible,
        gate_scope="all observed pairs plus paired-mean confidence; not future guarantee")


def reuse_break_even(native_cold_ms, native_hot_ms, opt_cold_ms, opt_hot_ms,
                     *, margin_ms=0.0):
    """Analytical reuse estimate from separately measured cold/hot full paths.

    Assume the SAME plan and weight version are valid for M calls. One call is
    cold and M-1 are hot for BOTH baseline and candidate. This is not a measured
    multi-microbatch run and is invalid across an optimizer weight update.
    """
    delta_cold=native_cold_ms-opt_cold_ms
    delta_hot=native_hot_ms-opt_hot_ms
    if delta_cold > margin_ms:
        reuse=1
    elif delta_hot <= margin_ms:
        reuse=None
    else:
        extra=delta_hot-delta_cold
        reuse=max(1, math.floor(extra/(delta_hot-margin_ms))+1)
        while delta_hot+(delta_cold-delta_hot)/reuse <= margin_ms:
            reuse+=1
    return dict(kind="analytical_from_measured_endpoints",
        cold_net_saving_ms=delta_cold,hot_net_saving_ms=delta_hot,
        incremental_cold_over_hot_cost_ms=delta_hot-delta_cold,
        strict_positive_min_same_version_reuses=reuse,
        average_saving_formula="delta_hot + (delta_cold - delta_hot) / M",
        conditions="unchanged weight version, unchanged replica mapping, one cold + M-1 hot calls for both methods")


def copied_slots(mapping, *, previous_mapping=None, same_weight_version=False):
    """Exact-slot cold/hot copy count; mapping must contain unique IDs per row.

    If an expert moves to another slot, a copy is conservatively required here.
    An implementation using local slot-to-slot copies should report those
    separately instead of claiming this value is actual NVLink bytes.
    """
    if previous_mapping is not None and len(previous_mapping)!=len(mapping):
        raise ValueError("rank count changed")
    copies=[]
    for r,row in enumerate(mapping):
        entries=[e for e in row if e>=0]
        if len(entries)!=len(set(entries)):
            raise ValueError("duplicate replica of an expert on one rank")
        for slot,e in enumerate(row):
            if e<0:
                continue
            cached=(same_weight_version and previous_mapping is not None and
                    slot<len(previous_mapping[r]) and previous_mapping[r][slot]==e)
            if not cached:
                copies.append((r,slot,e))
    return copies


def communication_accounting(*, baseline_remote_destinations,
        optimized_remote_destinations, baseline_mapping, optimized_mapping,
        hidden, token_element_bytes, expert_bytes,
        baseline_previous_mapping=None, optimized_previous_mapping=None,
        same_weight_version=False):
    """Payload bytes, not link-counter bytes or latency estimates.

    remote_destinations counts unique (original token,remote execution rank).
    BF16 dispatch and BF16 rank-reduced combine each send one hidden vector.
    This explicitly excludes expert-route metadata, router weights, padding,
    headers, and any backend that communicates a different combine layout.
    """
    basecopies=copied_slots(baseline_mapping,
        previous_mapping=baseline_previous_mapping,same_weight_version=same_weight_version)
    optcopies=copied_slots(optimized_mapping,
        previous_mapping=optimized_previous_mapping,same_weight_version=same_weight_version)
    b=sum(baseline_remote_destinations);o=sum(optimized_remote_destinations)
    unit=hidden*token_element_bytes
    saved_dispatch=(b-o)*unit
    saved_combine=saved_dispatch
    bw=len(basecopies)*expert_bytes;ow=len(optcopies)*expert_bytes
    return dict(remote_destinations_baseline=b,remote_destinations_optimized=o,
        baseline_copied_slots=basecopies,optimized_copied_slots=optcopies,
        baseline_expert_payload_bytes=bw,optimized_expert_payload_bytes=ow,
        incremental_expert_payload_bytes=ow-bw,
        saved_dispatch_payload_bytes=saved_dispatch,
        saved_combine_payload_bytes=saved_combine,
        net_saved_payload_bytes=saved_dispatch+saved_combine-(ow-bw),
        excludes="metadata, router weights, protocol overhead, padding, gradients",
        is_latency_or_model_speedup=False)


def normalize_benchmark(payload, *, metric="cuda_ms"):
    """Adapt benchmark.py's retained per-rank records without changing cohort.

    CUDA and host durations are analyzed separately. The host timer begins after
    the common barrier/device drain and ends after the completion event; it is
    a process-observed segment time, not a full model iteration.
    """
    names={"nativecoldserial":"native_cold_serial",
           "nativecoldearly":"native_cold_early","nativehot":"native_hot",
           "optcoldserial":"optimized_cold_serial",
           "optcoldearly":"optimized_cold_early","opthot":"optimized_hot"}
    meta=payload["metadata"]
    output={"schema_version":1,"status":payload.get("status","unknown"),
        "failures":payload.get("failures",[]),"metadata":{
            "physical_ranks":meta["world"],
            "timed_scope":"planner_weights_dispatch_combine",
            "metric":metric,
            "hot_weights_same_version_and_mapping":bool(
                meta.get("hot_exact_map_prefill_per_sample",False))},"layers":[]}
    for row in payload["layers"]:
        measurements=row["measurements"]
        missing=set(names.values())-set(measurements)
        if missing:
            raise ValueError(f"layer {row['layer']} missing full-path modes: {sorted(missing)}")
        lengths={len(measurements[name]) for name in names.values()}
        if len(lengths)!=1 or not lengths or next(iter(lengths))!=meta["samples"]:
            raise ValueError(f"layer {row['layer']} has incomplete/unequal sample cohorts")
        samples=[]
        for rep in range(meta["samples"]):
            times={name:[rank[metric] for rank in measurements[raw][rep]["per_rank"]]
                   for name,raw in names.items()}
            samples.append({"round":rep,"rank_times_ms":times})
        audit=bool(row.get("post_timing_checked",False)) and all(row["correctness"].values())
        output["layers"].append({"layer":row["layer"],"audit_pass":audit,"samples":samples})
    return output


def byte_accounting_from_benchmark(payload):
    """Use audited native/optimized copy COUNTS; preserve each baseline's cost."""
    metadata=payload["metadata"]
    result=[]
    for row in payload["layers"]:
        c=row["communication"]
        removed=sum(c["remote_before_by_rank"])-sum(c["remote_after_by_rank"])
        both=removed*2*metadata["hidden"]*2
        if both!=c["dispatch_combine_bytes_saved"]:
            raise ValueError("reported token-byte reduction disagrees with remote counts")
        base_n=sum(c["native_copies"]);opt_n=sum(c["optimized_copies"])
        extra=(opt_n-base_n)*metadata["expert_bytes"]
        result.append(dict(layer=row["layer"],
            baseline_copy_count=base_n,optimized_copy_count=opt_n,
            baseline_expert_bytes=base_n*metadata["expert_bytes"],
            optimized_expert_bytes=opt_n*metadata["expert_bytes"],
            incremental_expert_bytes=extra,saved_dispatch_combine_payload_bytes=both,
            cold_net_saved_payload_bytes=both-extra,
            excluded="metadata, route weights, padding, protocol overhead, gradient traffic",
            expert_bytes_is_model_shape_assumption=metadata.get("weight_shape_source")!="verified_model_config",
            byte_savings_do_not_prove_latency_savings=True))
    return result


def analyze(payload, *, guardband_ms=0.0, bootstrap_repeats=2000):
    meta=payload["metadata"]
    if meta["physical_ranks"]!=2:
        raise ValueError("expected a two-physical-rank experiment")
    if meta["timed_scope"]!="planner_weights_dispatch_combine":
        raise ValueError("component samples cannot certify full communication path")
    out={"schema_version":1,"scope":meta["timed_scope"],
         "source_failures":payload.get("failures",[]),"layers":[],
         "notice":"No GEMM/gradient/model end-to-end guarantee; gate includes all measured planning cost."}
    allcold=True
    allhot=bool(meta.get("hot_weights_same_version_and_mapping",False))
    for row in payload["layers"]:
        layer={"layer":row["layer"],"audit_pass":row["audit_pass"],"comparisons":[]}
        available=set.intersection(*(set(x["rank_times_ms"]) for x in row["samples"]))
        # Every candidate is tested against BOTH native schedules. Passing only
        # the slower serial baseline is insufficient for a positive gate.
        for candidate in ("optcoldserial","optcoldearly"):
            comparisons=[]
            for baseline in ("nativecoldserial","nativecoldearly"):
                c=paired_comparison(row["samples"],baseline,candidate,
                    guardband_ms=guardband_ms,bootstrap_repeats=bootstrap_repeats)
                comparisons.append(c);layer["comparisons"].append(c)
            layer[candidate+"_eligible"]=bool(row["audit_pass"]) and all(
                c["measured_positive_gate_pass"] for c in comparisons)
        coldpass=layer["optcoldserial_eligible"] or layer["optcoldearly_eligible"]
        allcold &= coldpass
        if {"nativehot","opthot"}<=available:
            c=paired_comparison(row["samples"],"nativehot","opthot",
                guardband_ms=guardband_ms,bootstrap_repeats=bootstrap_repeats)
            layer["comparisons"].append(c)
            layer["hot_eligible"]=(row["audit_pass"] and c["measured_positive_gate_pass"]
                and bool(meta.get("hot_weights_same_version_and_mapping",False)))
            allhot &= layer["hot_eligible"]
            native_cold=min(stats.median(rank_max(x["rank_times_ms"][mode]) for x in row["samples"])
                            for mode in ("nativecoldserial","nativecoldearly"))
            opt_cold=min(stats.median(rank_max(x["rank_times_ms"][mode]) for x in row["samples"])
                         for mode in ("optcoldserial","optcoldearly"))
            layer["reuse_analysis"]=reuse_break_even(native_cold,c["baseline_median_ms"],
                opt_cold,c["candidate_median_ms"],margin_ms=guardband_ms)
        else:
            allhot=False
        out["layers"].append(layer)
    complete=(bool(out["layers"]) and not out["source_failures"] and
              payload.get("status","complete")=="complete")
    out["cold_measured_gate_all_layers"]=complete and allcold
    out["hot_measured_gate_all_layers"]=complete and allhot
    out["future_positive_guaranteed"]=False
    return out


def analyze_benchmark(payload, *, guardband_ms=0.0, bootstrap_repeats=2000):
    result={"schema_version":1,"raw_status":payload.get("status"),
            "raw_failures":payload.get("failures",[]),"analysis":{},
            "scope":"actual two-rank planner+weights+dispatch+identity-combine; no GEMM",
            "bytes":byte_accounting_from_benchmark(payload)}
    for metric in ("cuda_ms","host_ms"):
        result["analysis"][metric]=analyze(normalize_benchmark(payload,metric=metric),
            guardband_ms=guardband_ms,bootstrap_repeats=bootstrap_repeats)
    # One actual schedule must qualify in both metrics; a CUDA win for serial
    # and a host win for early cannot be combined into a fictitious winner.
    qualified=[]
    cuda_layers=result["analysis"]["cuda_ms"]["layers"]
    host_layers=result["analysis"]["host_ms"]["layers"]
    for cuda,host in zip(cuda_layers,host_layers):
        assert cuda["layer"]==host["layer"]
        modes=[name for name in ("optcoldserial","optcoldearly")
               if cuda[name+"_eligible"] and host[name+"_eligible"]]
        qualified.append({"layer":cuda["layer"],"eligible_modes":modes})
    result["qualified_cold_modes_by_layer"]=qualified
    result["cold_measured_gate_pass"]=(payload.get("status")=="complete" and
        not payload.get("failures") and bool(qualified) and all(x["eligible_modes"] for x in qualified))
    result["hot_measured_gate_pass"]=all(v["hot_measured_gate_all_layers"]
                                          for v in result["analysis"].values())
    result["default_enable_dynamic_replication"]=result["cold_measured_gate_pass"]
    result["future_positive_guaranteed"]=False
    result["gating_caveat"]=("Post-planning fallback cannot refund its planning cost. "
        "This measured eligibility decision is not an online zero-overhead oracle.")
    return result


def self_check():
    s=[{"round":i,"rank_times_ms":{"b":[10,2],"o":[3,9]}} for i in range(10)]
    assert paired_comparison(s,"b","o")["median_delta_ms"]==1
    assert reuse_break_even(2,1,4,0.5)["strict_positive_min_same_version_reuses"]==6
    assert reuse_break_even(2,1,4,1)["strict_positive_min_same_version_reuses"] is None
    c=communication_accounting(baseline_remote_destinations=[10,20],
        optimized_remote_destinations=[5,15],baseline_mapping=[[2],[-1]],
        optimized_mapping=[[2],[0]],hidden=4,token_element_bytes=2,expert_bytes=100)
    assert c["incremental_expert_payload_bytes"]==100
    assert c["net_saved_payload_bytes"]==60
    assert len(copied_slots([[2],[-1]],previous_mapping=[[2],[-1]],same_weight_version=True))==0
    assert len(copied_slots([[2],[-1]],previous_mapping=[[2],[-1]],same_weight_version=False))==1
    print("Accounting self-check passed: slowest rank, incremental copies, versioned cache, reuse threshold")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--guardband-ms",type=float,default=0.0)
    p.add_argument("--bootstrap-repeats",type=int,default=2000)
    p.add_argument("--self-check",action="store_true")
    a=p.parse_args()
    if a.self_check:
        self_check();return
    if a.input is None or a.output is None:
        p.error("--input and --output required")
    raw=a.input.read_bytes()
    payload=json.loads(raw)
    method=analyze_benchmark if "world" in payload["metadata"] else analyze
    out=method(payload,guardband_ms=a.guardband_ms,bootstrap_repeats=a.bootstrap_repeats)
    out["source_sha256"]=hashlib.sha256(raw).hexdigest()
    out["source_path"]=str(a.input)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
