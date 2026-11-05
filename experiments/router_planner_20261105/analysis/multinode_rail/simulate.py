"""Exact rail traffic and a conservative chunk-scheduled MoE reference model.

This is CPU simulation, not a measured DeepEP/MoonEP or GPU latency model.
Dispatch and combine pipeline RDMA and NVLink within each phase. Global
barriers separate dispatch, compute and combine. A deterministic whole-chunk
FCFS scheduler reserves NIC/NVLink injection and receive ports together; it
does not model packet-level fair sharing, finite buffers, HBM or congestion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    hidden: int = 7168
    input_bytes_per_element: int = 2
    output_bytes_per_element: int = 2
    rdma_gbytes_per_second: float = 40.0
    nvlink_gbytes_per_second: float = 150.0
    rdma_startup_us: float = 2.0
    nvlink_startup_us: float = 0.5
    chunk_tokens: int = 128
    expert_padding: int = 128
    compute_pairs_per_second_per_gpu: float = 20_000_000.0
    compute_use_padding: bool = True
    planner_us: float = 0.0
    rail_window_us: float = 100.0

    def validate(self):
        positive = (self.hidden, self.input_bytes_per_element,
                    self.output_bytes_per_element, self.rdma_gbytes_per_second,
                    self.nvlink_gbytes_per_second, self.chunk_tokens,
                    self.expert_padding, self.compute_pairs_per_second_per_gpu,
                    self.rail_window_us)
        if any(x <= 0 for x in positive):
            raise ValueError("sizes, rates and window must be positive")
        if min(self.rdma_startup_us, self.nvlink_startup_us, self.planner_us) < 0:
            raise ValueError("latencies cannot be negative")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    return dict(total=float(values.sum()), mean=mean, maximum=float(values.max()),
                std_population=float(values.std()),
                max_over_mean=float(values.max() / mean) if mean else None,
                cv=float(values.std() / mean) if mean else None)


def traffic_counters(topk, owner, home_rank, nodes, gpus, config=Config()):
    """Exact payload-copy counts; axes are explicitly named in the result."""
    x, o = np.asarray(topk), np.asarray(owner)
    ranks, per_rank, _ = x.shape
    experts = len(home_rank)
    if o.shape != x.shape or ranks != nodes * gpus:
        raise ValueError("shape/topology mismatch")
    if np.any(o < 0) or np.any(o >= ranks) or np.any(x < 0) or np.any(x >= experts):
        raise ValueError("invalid expert or owner")
    rdma = np.zeros((nodes, gpus, nodes), np.int64)
    nvlink = np.zeros((nodes, gpus, gpus), np.int64)
    rank_copies = np.zeros((ranks, ranks), np.int64)
    pair_counts = np.bincount((o.astype(np.int64) * experts + x).ravel(),
                             minlength=ranks * experts).reshape(ranks, experts)
    fanout = np.zeros(nodes, np.int64)
    for src in range(ranks):
        source_node, rail = divmod(src, gpus)
        visited = (o[src, :, :, None] == np.arange(ranks)).any(axis=1)
        node_visited = visited.reshape(per_rank, nodes, gpus).any(axis=2)
        remote = node_visited.copy()
        remote[:, source_node] = False
        rdma[source_node, rail] = remote.sum(axis=0)
        fanout += np.bincount(remote.sum(axis=1), minlength=nodes)
        rank_copies[src] = visited.sum(axis=0)
        for dst in range(ranks):
            node, local = divmod(dst, gpus)
            if local != rail:
                nvlink[node, rail, local] += rank_copies[src, dst]
    padded = ((pair_counts + config.expert_padding - 1) // config.expert_padding
              * config.expert_padding)
    pair_loads = pair_counts.sum(axis=1)
    extras = ((pair_counts > 0) & (np.arange(ranks)[:, None] != home_rank[None])).sum(axis=1)
    input_bytes = config.hidden * config.input_bytes_per_element
    output_bytes = config.hidden * config.output_bytes_per_element
    tx, rx = rdma.sum(axis=2), rdma.sum(axis=0).T
    rail = rdma.sum(axis=(0, 2))
    max_port_copies = int(max(tx.max(), rx.max()))
    max_nv_port_copies = int(max(nvlink.sum(axis=2).max(), nvlink.sum(axis=1).max()))
    result = {
        "tokens": ranks * per_rank, "pairs": int(x.size), "top_k": x.shape[-1],
        "rdma_copies_src_node_rail_dst_node": rdma.tolist(),
        "nvlink_copies_node_src_local_dst_local": nvlink.tolist(),
        "combine_rdma_contract": "one node-local reduced output per token/remote node; reverse dispatch edges",
        "combine_nvlink_contract": "one rank-local reduced output per token/execution rank; reverse dispatch NVLink edges",
        "rail_rdma_token_copies": rail.tolist(), "rail_statistics": distribution(rail),
        "rdma_token_copies": int(rdma.sum()), "nvlink_token_copies": int(nvlink.sum()),
        "rdma_dispatch_payload_bytes": int(rdma.sum()) * input_bytes,
        "rdma_combine_payload_bytes": int(rdma.sum()) * output_bytes,
        "nvlink_dispatch_payload_bytes": int(nvlink.sum()) * input_bytes,
        "nvlink_combine_payload_bytes": int(nvlink.sum()) * output_bytes,
        "nic_tx_token_copies_node_rail": tx.tolist(),
        "nic_rx_token_copies_node_rail": rx.tolist(),
        "nic_tx_statistics": distribution(tx), "nic_rx_statistics": distribution(rx),
        "max_nic_port_token_copies": max_port_copies,
        "max_nic_dispatch_payload_bytes": max_port_copies * input_bytes,
        "max_nvlink_port_token_copies": max_nv_port_copies,
        "remote_node_fanout_histogram": fanout.tolist(),
        "rank_destination_copies_including_source": int(rank_copies.sum()),
        "pair_loads_per_rank": pair_loads.tolist(),
        "pair_loads_per_node": pair_loads.reshape(nodes, gpus).sum(axis=1).tolist(),
        "pair_load_statistics": distribution(pair_loads),
        "pair_counts_rank_expert": pair_counts.tolist(),
        "padded_pair_counts_rank_expert": padded.tolist(),
        "padded_pair_loads_per_rank": padded.sum(axis=1).tolist(),
        "padding_overhead_pairs": int(padded.sum() - pair_counts.sum()),
        "active_extra_experts_per_rank": extras.tolist(),
        "rdma_dispatch_port_service_lower_bound_us": max_port_copies * input_bytes / (config.rdma_gbytes_per_second * 1e3),
        "nvlink_dispatch_port_service_lower_bound_us": max_nv_port_copies * input_bytes / (config.nvlink_gbytes_per_second * 1e3),
    }
    assert int(pair_counts.sum()) == x.size
    return result


@dataclass
class Chunk:
    source: int
    node: int
    ordinal: int
    tokens: np.ndarray
    destination_counts: tuple


def build_chunks(owner, nodes, gpus, chunk_tokens):
    """Pack source-rank/destination-node lists in source token order."""
    chunks = []
    ranks = nodes * gpus
    for source in range(ranks):
        visited = (owner[source, :, :, None] == np.arange(ranks)).any(axis=1)
        for node in range(nodes):
            local_visited = visited[:, node * gpus:(node + 1) * gpus]
            ids = np.flatnonzero(local_visited.any(axis=1))
            for ordinal, begin in enumerate(range(0, len(ids), chunk_tokens)):
                token_ids = ids[begin:begin + chunk_tokens]
                counts = local_visited[token_ids].sum(axis=0)
                destinations = tuple((int(q), int(n)) for q, n in enumerate(counts) if n)
                chunks.append(Chunk(source, node, ordinal, token_ids, destinations))
    # Round-robin chunk admission prevents a source's entire tensor going first.
    return sorted(chunks, key=lambda c: (c.ordinal, c.source, c.node))


class PortScheduler:
    """Conservative nonpreemptive list scheduling, with shared endpoint ports."""
    def __init__(self, config):
        self.config = config
        self.free = defaultdict(float)
        self.records = []

    def send(self, kind, node_a, local_a, node_b, local_b, ready, byte_count):
        if kind == "rdma":
            if local_a != local_b or node_a == node_b:
                raise ValueError("RDMA must preserve the source rail and cross nodes")
            bandwidth = self.config.rdma_gbytes_per_second * 1e3
            startup = self.config.rdma_startup_us
        elif kind == "nvlink":
            if node_a != node_b or local_a == local_b:
                raise ValueError("NVLink must remain within one node and change rank")
            bandwidth = self.config.nvlink_gbytes_per_second * 1e3
            startup = self.config.nvlink_startup_us
        else:
            raise ValueError(kind)
        resources = ((kind + "_tx", node_a, local_a), (kind + "_rx", node_b, local_b))
        start = max(ready, *(self.free[r] for r in resources))
        end = start + startup + byte_count / bandwidth
        for resource in resources:
            self.free[resource] = end
        self.records.append(dict(kind=kind, source_node=node_a, source_local=local_a,
                                 destination_node=node_b, destination_local=local_b,
                                 ready_us=float(ready), start_us=float(start), end_us=float(end),
                                 bytes=int(byte_count), resources=resources))
        return end

    def summary(self, gpus):
        duration = max((r["end_us"] for r in self.records), default=0.0)
        queues = defaultdict(list)
        busy = defaultdict(float)
        for r in self.records:
            for resource in r["resources"]:
                busy[resource] += r["end_us"] - r["start_us"]
                if r["start_us"] > r["ready_us"]:
                    queues[resource].append((r["ready_us"], 1, r["bytes"]))
                    queues[resource].append((r["start_us"], -1, -r["bytes"]))
        ports = {}
        for resource in sorted(busy):
            pending = pending_bytes = peak = peak_bytes = 0
            # Dequeue before enqueue on equal timestamps; zero waits excluded.
            for _, delta, byte_delta in sorted(queues[resource]):
                pending += delta
                pending_bytes += byte_delta
                peak, peak_bytes = max(peak, pending), max(peak_bytes, pending_bytes)
            assert pending == pending_bytes == 0
            ports["/".join(map(str, resource))] = dict(
                busy_us=busy[resource], utilization=busy[resource] / duration if duration else 0,
                peak_waiting_chunks=peak, peak_waiting_payload_bytes=peak_bytes)
        window = self.config.rail_window_us
        bins = np.zeros((max(1, int(np.ceil(duration / window))), gpus), dtype=np.float64)
        for r in self.records:
            if r["kind"] != "rdma":
                continue
            begin, end = r["start_us"], r["end_us"]
            for b in range(int(begin // window), min(len(bins), int(np.ceil(end / window)))):
                overlap = max(0.0, min(end, (b + 1) * window) - max(begin, b * window))
                bins[b, r["source_local"]] += r["bytes"] * overlap / (end - begin)
        return {
            "makespan_us": duration, "transfer_chunks": len(self.records),
            "rdma_chunks": sum(r["kind"] == "rdma" for r in self.records),
            "nvlink_chunks": sum(r["kind"] == "nvlink" for r in self.records),
            "rdma_payload_bytes": sum(r["bytes"] for r in self.records if r["kind"] == "rdma"),
            "nvlink_payload_bytes": sum(r["bytes"] for r in self.records if r["kind"] == "nvlink"),
            "mean_chunk_wait_us": float(np.mean([r["start_us"] - r["ready_us"] for r in self.records])) if self.records else 0.0,
            "ports": ports,
            "rail_window_us": window,
            "rail_window_semantics": "payload bytes spread uniformly across each chunk service interval, including its assumed startup; time starts at this phase",
            "rail_payload_bytes_by_window": bins.tolist(),
        }


def simulate_chunks(owner, nodes, gpus, config=Config()):
    """Return dispatch and reverse-combine schedules, both relative to zero."""
    chunks = build_chunks(owner, nodes, gpus, config.chunk_tokens)
    dispatch = PortScheduler(config)
    combine = PortScheduler(config)
    in_bytes = config.hidden * config.input_bytes_per_element
    out_bytes = config.hidden * config.output_bytes_per_element
    heap = []
    serial = 0
    # Events: arrival at the destination relay, then rank-local forwarding.
    for cid, c in enumerate(chunks):
        source_node, rail = divmod(c.source, gpus)
        if c.node != source_node:
            heapq.heappush(heap, (0.0, serial, "rdma", cid, -1, len(c.tokens)))
            serial += 1
        else:
            for local, count in c.destination_counts:
                if local != rail:
                    heapq.heappush(heap, (0.0, serial, "nvlink", cid, local, count))
                    serial += 1
    while heap:
        ready, _, kind, cid, local, count = heapq.heappop(heap)
        c = chunks[cid]
        source_node, rail = divmod(c.source, gpus)
        if kind == "rdma":
            end = dispatch.send(kind, source_node, rail, c.node, rail, ready, count * in_bytes)
            for local, count in c.destination_counts:
                if local != rail:
                    heapq.heappush(heap, (end, serial, "nvlink", cid, local, count))
                    serial += 1
        else:
            dispatch.send(kind, c.node, rail, c.node, local, ready, count * in_bytes)

    # All ranks start combine after the global GEMM barrier. At a destination
    # node, the complete chunk waits for every contributing rank's local sum.
    ready_at_relay = np.zeros(len(chunks), dtype=np.float64)
    for cid, c in enumerate(chunks):
        _, rail = divmod(c.source, gpus)
        for local, count in c.destination_counts:
            if local != rail:
                end = combine.send("nvlink", c.node, local, c.node, rail, 0.0, count * out_bytes)
                ready_at_relay[cid] = max(ready_at_relay[cid], end)
    completion = np.zeros(owner.shape[:2], dtype=np.float64)
    for cid in sorted(range(len(chunks)), key=lambda i: (ready_at_relay[i], i)):
        c = chunks[cid]
        source_node, rail = divmod(c.source, gpus)
        end = float(ready_at_relay[cid])
        if c.node != source_node:
            end = combine.send("rdma", c.node, rail, source_node, rail, end, len(c.tokens) * out_bytes)
        completion[c.source, c.tokens] = np.maximum(completion[c.source, c.tokens], end)
    return dispatch, combine, completion


def evaluate(arrays, metadata, config=Config()):
    config.validate()
    if metadata.get("legal_plan") is False:
        raise ValueError("scenario reports an illegal seed plan")
    topk, owner, home = arrays["topk"], arrays["owner"], arrays["home_rank"]
    nodes, gpus = metadata["topology"]["nodes"], metadata["topology"]["gpus_per_node"]
    counts = traffic_counters(topk, owner, home, nodes, gpus, config)
    # Validate residency separately from pair execution. Copies are eligibility,
    # never an instruction to execute a pair at every resident location.
    support = np.arange(nodes * gpus)[:, None] == home[None]
    if "experts_to_copy" in arrays:
        for rank, copies in enumerate(arrays["experts_to_copy"]):
            support[rank, copies[copies >= 0]] = True
    if not np.all(support[owner, topk]):
        raise ValueError("execution assigned to a nonresident expert")
    dispatch, combine, completion = simulate_chunks(owner, nodes, gpus, config)
    d, c = dispatch.summary(gpus), combine.summary(gpus)
    assert d["rdma_payload_bytes"] == counts["rdma_dispatch_payload_bytes"]
    assert d["nvlink_payload_bytes"] == counts["nvlink_dispatch_payload_bytes"]
    assert c["rdma_payload_bytes"] == counts["rdma_combine_payload_bytes"]
    assert c["nvlink_payload_bytes"] == counts["nvlink_combine_payload_bytes"]
    field = "padded_pair_loads_per_rank" if config.compute_use_padding else "pair_loads_per_rank"
    compute_per_rank = np.array(counts[field]) / config.compute_pairs_per_second_per_gpu * 1e6
    compute_us = float(compute_per_rank.max())
    before_combine = config.planner_us + d["makespan_us"] + compute_us
    completion += before_combine
    layer_us = before_combine + c["makespan_us"]
    assert np.isclose(float(completion.max()), layer_us)
    return {
        "schema_version": 1, "scenario": metadata["scenario"],
        "provenance": metadata["provenance"], "topology": metadata["topology"],
        "scope": "uncalibrated CPU counterfactual; exact payload counts, assumed chunk-service and compute timings",
        "model": "global barriers: planner -> pipelined dispatch -> parallel per-rank GEMM -> pipelined combine",
        "config": asdict(config),
        "parameter_provenance": "ALL rates, startup costs, hidden size, dtypes, padding, chunk size and compute rates are illustrative assumptions, not measured hardware",
        "dtype_assumption": "BF16 dispatch and BF16 combine at default 2 bytes/element; no payload values materialized",
        "planner_status": "excluded_unmeasured" if config.planner_us == 0 else "user_supplied_assumed_latency_not_measured",
        "assumptions": [
            "Separate nonblocking switch per rail; shared full-duplex NIC Tx/Rx ports are the only RDMA fabric constraints.",
            "NVLink is a nonblocking per-node fabric limited by per-GPU injection/ejection ports; no individual-link routing.",
            "Whole-chunk nonpreemptive FCFS reservations, round-robin initial chunk admission; scheduler can introduce head-of-line idle time.",
            "Source local rank fixes every dispatch RDMA rail; no source-side NVLink load-balancing hop.",
            "At most one payload per token/remote node and one per token/destination rank after relay; combine reverses this with node-local partial reduction.",
            "Global dispatch/GEMM/combine barriers are conservative scheduling choices; these timings are not a prediction of a production pipelined kernel.",
            "All routes are ready at phase start; full chunk waits for all its contributions before reverse RDMA.",
        ],
        "excluded_costs": ["measured planner and metadata communication", "expert weight prefetch/migration",
                           "gradient/optimizer traffic", "protocol and metadata bytes", "HBM contention",
                           "finite buffering/backpressure", "switch oversubscription", "reduction arithmetic",
                           "router and other transformer-layer operations"],
        "traffic": counts, "dispatch": d, "combine": c,
        "latency_us": {
            "planner": config.planner_us, "dispatch": d["makespan_us"],
            "compute": compute_us, "compute_per_rank": compute_per_rank.tolist(),
            "combine": c["makespan_us"], "modeled_moe_layer": layer_us,
            "token_completion_quantiles": {str(q): float(np.percentile(completion, q)) for q in (50, 95, 99, 100)},
        },
        "checks": {"resident_execution": True, "pair_conservation": True,
                   "dispatch_counter_schedule_bytes_match": True,
                   "combine_counter_schedule_bytes_match": True,
                   "rdma_source_rail_invariant": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/multinode_rail/scenarios")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/multinode_rail/simulation")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--rdma-gbps", type=float, default=40.0, help="decimal GB/s per NIC direction, not Gbit/s")
    parser.add_argument("--nvlink-gbps", type=float, default=150.0, help="decimal GB/s per GPU port direction, not Gbit/s")
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--planner-us", type=float, default=0.0)
    parser.add_argument("--compute-pairs-per-second", type=float, default=20_000_000.0)
    parser.add_argument("--compute-without-padding", action="store_true")
    args = parser.parse_args()
    config = Config(rdma_gbytes_per_second=args.rdma_gbps,
                    nvlink_gbytes_per_second=args.nvlink_gbps,
                    chunk_tokens=args.chunk_tokens, planner_us=args.planner_us,
                    compute_pairs_per_second_per_gpu=args.compute_pairs_per_second,
                    compute_use_padding=not args.compute_without_padding)
    config.validate()
    names = args.scenarios or [x["scenario"] for x in json.loads((args.input / "manifest.json").read_text())["scenarios"]]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        start = time.perf_counter()
        path = args.input / (name + ".npz")
        metadata = json.loads((args.input / (name + ".json")).read_text())
        with np.load(path, allow_pickle=False) as arrays:
            result = evaluate(arrays, metadata, config)
        result["input_npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["simulator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["cpu_simulation_wall_seconds_not_planner_time"] = time.perf_counter() - start
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        row = dict(scenario=name, rail_max_over_mean=result["traffic"]["rail_statistics"]["max_over_mean"],
                   rail_cv=result["traffic"]["rail_statistics"]["cv"],
                   rdma_copies=result["traffic"]["rdma_token_copies"],
                   nvlink_copies=result["traffic"]["nvlink_token_copies"],
                   max_nic_port_copies=result["traffic"]["max_nic_port_token_copies"],
                   latency_us=result["latency_us"], output_file=name + ".json",
                   input_npz_sha256=result["input_npz_sha256"])
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "latency_us"} | {
            "modeled_moe_layer_us": row["latency_us"]["modeled_moe_layer"],
            "cpu_simulation_seconds": result["cpu_simulation_wall_seconds_not_planner_time"]}), flush=True)
    summary = {"scope": "uncalibrated assumed-service model; not measured hardware or planner performance",
               "config": asdict(config), "planner_status": "excluded_unmeasured" if config.planner_us == 0 else "assumed",
               "rows": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
