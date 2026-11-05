"""Reuse seed chunk membership and port order after edge-deleting planning.

This is an alternative metadata/scheduling policy, not a claim about DeepEP's
default kernel. Every original chunk keeps its order and token membership;
masked compaction drops obsolete token/node/rank copies within that chunk.
Each shared port retains its original order of surviving operations. Therefore
deleting payload cannot introduce the FCFS repacking regressions of rebuilding
an entirely new chunk stream. GPU metadata construction cost is unmeasured.
"""
from __future__ import annotations

import heapq
import numpy as np

from simulate import Config, PortScheduler, build_chunks, evaluate


def _tagged_seed_schedules(chunks, gpus, config):
    """Exactly simulate.py's call order, with original chunk IDs attached."""
    dispatch, combine = PortScheduler(config), PortScheduler(config)
    input_bytes = config.hidden * config.input_bytes_per_element
    output_bytes = config.hidden * config.output_bytes_per_element
    events, serial = [], 0
    for cid, chunk in enumerate(chunks):
        source_node, rail = divmod(chunk.source, gpus)
        if chunk.node != source_node:
            heapq.heappush(events, (0.0, serial, "rdma", cid, -1, len(chunk.tokens)))
            serial += 1
        else:
            for local, count in chunk.destination_counts:
                if local != rail:
                    heapq.heappush(events, (0.0, serial, "nvlink", cid, local, count))
                    serial += 1
    while events:
        ready, _, kind, cid, local, count = heapq.heappop(events)
        chunk = chunks[cid]
        source_node, rail = divmod(chunk.source, gpus)
        if kind == "rdma":
            end = dispatch.send(kind, source_node, rail, chunk.node, rail,
                                ready, count*input_bytes)
            dispatch.records[-1]["chunk_id"] = cid
            for target, target_count in chunk.destination_counts:
                if target != rail:
                    heapq.heappush(events, (end, serial, "nvlink", cid, target, target_count))
                    serial += 1
        else:
            dispatch.send(kind, chunk.node, rail, chunk.node, local, ready,
                          count*input_bytes)
            dispatch.records[-1]["chunk_id"] = cid
    ready_at_relay = np.zeros(len(chunks))
    for cid, chunk in enumerate(chunks):
        _, rail = divmod(chunk.source, gpus)
        for local, count in chunk.destination_counts:
            if local != rail:
                end = combine.send("nvlink", chunk.node, local, chunk.node,
                                   rail, 0.0, count*output_bytes)
                combine.records[-1]["chunk_id"] = cid
                ready_at_relay[cid] = max(ready_at_relay[cid], end)
    for cid in sorted(range(len(chunks)), key=lambda i: (ready_at_relay[i], i)):
        chunk = chunks[cid]
        source_node, rail = divmod(chunk.source, gpus)
        if chunk.node != source_node:
            combine.send("rdma", chunk.node, rail, source_node, rail,
                         ready_at_relay[cid], len(chunk.tokens)*output_bytes)
            combine.records[-1]["chunk_id"] = cid
    return dispatch, combine


def _project(chunks, seed_owner, owner, nodes, gpus):
    """Validate per-token directed-edge subset, then project original chunks."""
    ranks = nodes*gpus
    if seed_owner.shape != owner.shape or owner.shape[0] != ranks:
        raise ValueError("owner/topology mismatch")
    if np.any(owner < 0) or np.any(owner >= ranks):
        raise ValueError("invalid owner")
    for source in range(ranks):
        source_node, rail = divmod(source, gpus)
        old = (seed_owner[source, :, :, None] == np.arange(ranks)).any(1)
        new = (owner[source, :, :, None] == np.arange(ranks)).any(1)
        old_nodes = old.reshape(-1, nodes, gpus).any(2)
        new_nodes = new.reshape(-1, nodes, gpus).any(2)
        remote = np.arange(nodes) != source_node
        if np.any(new_nodes[:, remote] & ~old_nodes[:, remote]):
            raise ValueError("reuse requires each token's RDMA edges to be a subset")
        nv_dest = np.arange(ranks) % gpus != rail
        if np.any(new[:, nv_dest] & ~old[:, nv_dest]):
            raise ValueError("reuse requires each token's NVLink edges to be a subset")
    counts, live_tokens = [], []
    for chunk in chunks:
        rows = owner[chunk.source, chunk.tokens]
        local_destinations = np.arange(chunk.node*gpus, (chunk.node+1)*gpus)
        visited = (rows[:, :, None] == local_destinations).any(1)
        counts.append(visited.sum(0))
        live_tokens.append(chunk.tokens[visited.any(1)])
    return np.array(counts, dtype=np.int64), live_tokens


def _replay(original, chunks, projected_counts, live_tokens, gpus, config, reverse):
    scheduler = PortScheduler(config)
    completion = np.zeros(len(chunks))
    relay_ready = np.zeros(len(chunks))
    payload_bytes = config.hidden * (config.output_bytes_per_element if reverse
                                    else config.input_bytes_per_element)
    for record in original.records:
        cid = record["chunk_id"]
        chunk = chunks[cid]
        source_node, _ = divmod(chunk.source, gpus)
        if record["kind"] == "rdma":
            count = len(live_tokens[cid])
            ready = relay_ready[cid] if reverse else 0.0
        else:
            local = record["source_local"] if reverse else record["destination_local"]
            count = int(projected_counts[cid, local])
            ready = 0.0 if reverse or chunk.node == source_node else relay_ready[cid]
        if count == 0:
            continue
        if count*payload_bytes > record["bytes"]:
            raise AssertionError("projected operation grew")
        end = scheduler.send(record["kind"], record["source_node"],
                             record["source_local"], record["destination_node"],
                             record["destination_local"], ready, count*payload_bytes)
        scheduler.records[-1]["chunk_id"] = cid
        # The saved reservation order is a topological order of per-port and
        # relay dependencies, so shorter/deleted operations cannot finish later.
        if end > record["end_us"] + 1e-7:
            raise AssertionError("reused operation completed later than seed")
        completion[cid] = max(completion[cid], end)
        if (reverse and record["kind"] == "nvlink") or (not reverse and record["kind"] == "rdma"):
            relay_ready[cid] = max(relay_ready[cid], end)
    return scheduler, completion


def _copies(support, home):
    nonhome = np.asarray(support) & (np.arange(len(support))[:, None] != home[None, :])
    result = np.full((len(support), int(nonhome.sum(1).max())), -1, np.int32)
    for rank in range(len(support)):
        ids = np.flatnonzero(nonhome[rank])
        result[rank, :len(ids)] = ids
    return result


def evaluate_reused_schedule(topk, seed_owner, owner, home, support, meta, config=Config()):
    """Return evaluate's schema using original seed chunks and port order.

    The payload-edge subset contract is per token, not merely aggregate. A free
    execution at the source GPU or existing RDMA relay is permitted even when
    that rank did not execute the token previously. Neither adds a data edge.
    """
    nodes, gpus = meta["topology"]["nodes"], meta["topology"]["gpus_per_node"]
    config.validate()
    chunks = build_chunks(seed_owner, nodes, gpus, config.chunk_tokens)
    projected, live = _project(chunks, np.asarray(seed_owner), np.asarray(owner), nodes, gpus)
    seed_dispatch, seed_combine = _tagged_seed_schedules(chunks, gpus, config)
    dispatch, _ = _replay(seed_dispatch, chunks, projected, live, gpus, config, False)
    combine, chunk_completion = _replay(seed_combine, chunks, projected, live, gpus, config, True)
    # Reuse evaluate's validation, exact counters, padding and metadata schema.
    # Its fresh schedule is discarded; this is CPU reference scaffolding, not
    # a prescription to perform both schedules inside a GPU planner.
    result = evaluate(dict(topk=topk, owner=owner, home_rank=home,
                           experts_to_copy=_copies(support, np.asarray(home))), meta, config)
    d, c = dispatch.summary(gpus), combine.summary(gpus)
    sd, sc = seed_dispatch.summary(gpus), seed_combine.summary(gpus)
    counts = result["traffic"]
    for phase, stats in (("dispatch", d), ("combine", c)):
        assert stats["rdma_payload_bytes"] == counts[f"rdma_{phase}_payload_bytes"]
        assert stats["nvlink_payload_bytes"] == counts[f"nvlink_{phase}_payload_bytes"]
    assert d["makespan_us"] <= sd["makespan_us"] + 1e-7
    assert c["makespan_us"] <= sc["makespan_us"] + 1e-7
    before_combine = config.planner_us + d["makespan_us"] + result["latency_us"]["compute"]
    completion = np.zeros(owner.shape[:2])
    for cid, chunk in enumerate(chunks):
        completion[chunk.source, live[cid]] = np.maximum(
            completion[chunk.source, live[cid]], chunk_completion[cid])
    completion += before_combine
    result["dispatch"], result["combine"] = d, c
    result["latency_us"].update(
        dispatch=d["makespan_us"], combine=c["makespan_us"],
        modeled_moe_layer=before_combine+c["makespan_us"],
        token_completion_quantiles={str(q): float(np.percentile(completion, q)) for q in (50, 95, 99, 100)})
    assert np.isclose(completion.max(), result["latency_us"]["modeled_moe_layer"])
    result["model"] = "global phase barriers; original seed chunk membership and per-port operation order reused with masked compaction"
    result["schedule_policy"] = {
        "name": "reuse_seed_chunk_and_port_order",
        "implementation_status": "CPU reference; metadata compaction/order reuse is proposed kernel integration, GPU cost unmeasured",
        "seed_dispatch_us": sd["makespan_us"], "seed_combine_us": sc["makespan_us"],
        "seed_chunks": len(chunks), "active_projected_chunks": sum(bool(len(ids)) for ids in live),
        "seed_dispatch_operations": len(seed_dispatch.records), "seed_combine_operations": len(seed_combine.records),
        "dispatch_operations": len(dispatch.records), "combine_operations": len(combine.records),
        "contract": "No new per-token RDMA or NVLink payload edge; retained chunk IDs and original per-port order; zero-byte operations removed. Phase monotonicity excludes GEMM padding and planner overhead.",
    }
    result["assumptions"].append("This alternative policy retains the seed's chunk membership and per-port operation order after planning; it is not the default DeepEP kernel schedule.")
    result["checks"].update(per_token_payload_edge_subset=True, seed_order_reuse=True,
                            projected_schedule_exact_payload=True,
                            dispatch_combine_latency_no_greater_than_seed=True)
    return result
