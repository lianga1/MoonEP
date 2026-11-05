"""Exact expert-copy bytes and illustrative weight/gradient port schedules.

Weights originate at each expert's home GPU. A remote-node copy traverses the
home rail once, then fans out over destination-node NVLink. Gradients reduce
onto that same relay and reverse the path. This models communication payloads,
not reduction arithmetic or a complete training iteration. All rates are the
same explicit assumptions as simulate.py, not measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import numpy as np

from simulate import Config, PortScheduler, distribution


@dataclass(frozen=True)
class WeightConfig:
    intermediate: int = 2048
    matrices: int = 3
    bytes_per_element: int = 2
    gradient_bytes_per_element: int = 2
    chunk_bytes: int = 1 << 20

    def validate(self):
        if min(self.intermediate, self.matrices, self.bytes_per_element,
               self.chunk_bytes) <= 0 or self.gradient_bytes_per_element < 0:
            raise ValueError("weight dimensions/chunk must be positive; gradient dtype bytes nonnegative")


def active_residency(topk, owner, home_rank):
    x, o, h = np.asarray(topk), np.asarray(owner), np.asarray(home_rank)
    if x.ndim != 3 or o.shape != x.shape or h.ndim != 1 or len(h) == 0:
        raise ValueError("invalid topk/owner/home shapes")
    ranks, experts = x.shape[0], len(h)
    if (np.any(x < 0) or np.any(x >= experts) or np.any(o < 0) or
            np.any(o >= ranks) or np.any(h < 0) or np.any(h >= ranks)):
        raise ValueError("expert or rank outside topology")
    counts = np.bincount((o.astype(np.int64) * experts + x).ravel(),
                         minlength=ranks * experts).reshape(ranks, experts)
    return counts > 0


def _copy_groups(resident, home, nodes, gpus):
    """One group per (expert,target node), containing nonhome target ranks."""
    groups = []
    for expert, source in enumerate(home):
        for node in range(nodes):
            targets = np.flatnonzero(resident[node*gpus:(node+1)*gpus, expert])
            targets = tuple(int(local) for local in targets if node*gpus+local != source)
            if targets:
                groups.append((int(expert), int(source), int(node), targets))
    return groups


def _counters(groups, per_expert_bytes, nodes, gpus, reverse=False):
    rdma = np.zeros((nodes, gpus, nodes), np.int64)
    nvlink = np.zeros((nodes, gpus, gpus), np.int64)
    for _, source, node, targets in groups:
        source_node, rail = divmod(source, gpus)
        if node != source_node:
            a, b = (node, source_node) if reverse else (source_node, node)
            rdma[a, rail, b] += per_expert_bytes
        for local in targets:
            if local != rail:
                a, b = (local, rail) if reverse else (rail, local)
                nvlink[node, a, b] += per_expert_bytes
    tx, rx = rdma.sum(2), rdma.sum(0).T
    nvtx, nvrx = nvlink.sum(2), nvlink.sum(1)
    rails = rdma.sum((0, 2))
    return {
        "rdma_bytes_src_node_rail_dst_node": rdma.tolist(),
        "nvlink_bytes_node_src_local_dst_local": nvlink.tolist(),
        "rdma_payload_bytes": int(rdma.sum()), "nvlink_payload_bytes": int(nvlink.sum()),
        "nic_tx_bytes_node_rail": tx.tolist(), "nic_rx_bytes_node_rail": rx.tolist(),
        "nvlink_tx_bytes_node_local": nvtx.tolist(), "nvlink_rx_bytes_node_local": nvrx.tolist(),
        "max_nic_port_bytes": int(max(tx.max(), rx.max())),
        "max_nvlink_port_bytes": int(max(nvtx.max(), nvrx.max())),
        "rail_rdma_bytes": rails.tolist(), "rail_statistics": distribution(rails),
    }


def _schedule(groups, per_expert_bytes, nodes, gpus, config, chunk_bytes, reverse=False):
    scheduler = PortScheduler(config)
    if not per_expert_bytes or not groups:
        return scheduler
    events, serial = [], 0
    # Interleave each expert's chunks; all tensor data is ready at phase start.
    for offset in range(0, per_expert_bytes, chunk_bytes):
        size = min(chunk_bytes, per_expert_bytes-offset)
        for gid, (_, source, node, targets) in enumerate(groups):
            source_node, rail = divmod(source, gpus)
            if not reverse and node != source_node:
                heapq.heappush(events, (0.0, serial, "rdma", gid, -1, size))
                serial += 1
            elif not reverse:
                for local in targets:
                    if local != rail:
                        heapq.heappush(events, (0.0, serial, "nvlink", gid, local, size))
                        serial += 1
            else:
                # Reverse gradient gathers all rank partials for this chunk.
                ready = 0.0
                for local in targets:
                    if local != rail:
                        end = scheduler.send("nvlink", node, local, node, rail, 0.0, size)
                        ready = max(ready, end)
                if node != source_node:
                    heapq.heappush(events, (ready, serial, "rdma", gid, -1, size))
                    serial += 1
    while events:
        ready, _, kind, gid, local, size = heapq.heappop(events)
        _, source, node, targets = groups[gid]
        source_node, rail = divmod(source, gpus)
        if kind == "rdma":
            a, b = (node, source_node) if reverse else (source_node, node)
            end = scheduler.send("rdma", a, rail, b, rail, ready, size)
            if not reverse:
                for target in targets:
                    if target != rail:
                        heapq.heappush(events, (end, serial, "nvlink", gid, target, size))
                        serial += 1
        else:
            scheduler.send("nvlink", node, rail, node, local, ready, size)
    return scheduler


def _phase(groups, nbytes, nodes, gpus, config, weight_config, reverse=False):
    counts = _counters(groups, nbytes, nodes, gpus, reverse)
    scheduler = _schedule(groups, nbytes, nodes, gpus, config,
                          weight_config.chunk_bytes, reverse)
    summary = scheduler.summary(gpus)
    assert counts["rdma_payload_bytes"] == summary["rdma_payload_bytes"]
    assert counts["nvlink_payload_bytes"] == summary["nvlink_payload_bytes"]
    # Return compact port summaries rather than large per-window scheduler bins.
    summary.pop("rail_payload_bytes_by_window")
    counts["payload_port_lower_bound_us"] = max(
        counts["max_nic_port_bytes"] / (config.rdma_gbytes_per_second*1e3),
        counts["max_nvlink_port_bytes"] / (config.nvlink_gbytes_per_second*1e3))
    counts["chunk_service_port_lower_bound_us"] = max(
        (p["busy_us"] for p in summary["ports"].values()), default=0.0)
    return {"traffic": counts, "schedule": summary}


def evaluate_weights(topk, owner, home_rank, nodes, gpus, config=Config(),
                     weight_config=WeightConfig(), initial_resident=None):
    """Cost demanded replicas only, never unused reserved expert slots.

    None starts from current home weights only. An explicit initial_resident
    boolean array represents already-present *current-version* weights; callers
    must not carry it across optimizer updates without a refresh. Sources remain
    home GPUs even if another warm cached copy exists. Gradients cover every
    active nonhome copy, including copies whose weights were initially warm.
    """
    config.validate()
    weight_config.validate()
    active = active_residency(topk, owner, home_rank)
    if active.shape[0] != nodes*gpus:
        raise ValueError("topology mismatch")
    home = np.asarray(home_rank)
    home_resident = np.arange(nodes*gpus)[:, None] == home[None, :]
    if initial_resident is None:
        initial = home_resident
        regime = "cold_current_version_home_only"
    else:
        initial = np.asarray(initial_resident)
        if initial.shape != active.shape or initial.dtype != np.bool_:
            raise ValueError("initial_resident must be a boolean [rank, expert] array")
        if not np.all(initial[home_resident]):
            raise ValueError("current home weights must be initially resident")
        regime = "explicit_current_version_warm_residency"
    missing = active & ~initial
    extras = active & ~home_resident
    parameters = config.hidden * weight_config.intermediate * weight_config.matrices
    weight_bytes = parameters * weight_config.bytes_per_element
    gradient_bytes = parameters * weight_config.gradient_bytes_per_element
    forward_groups = _copy_groups(missing, home, nodes, gpus)
    gradient_groups = _copy_groups(extras, home, nodes, gpus)
    forward = _phase(forward_groups, weight_bytes, nodes, gpus, config, weight_config)
    gradient = _phase(gradient_groups, gradient_bytes, nodes, gpus, config, weight_config, reverse=True)
    return {
        "scope": "exact replica payloads plus uncalibrated shared-port chunk schedules; no measured GPU latency",
        "initial_regime": regime, "weight_config": asdict(weight_config),
        "parameters_per_expert": parameters, "weight_bytes_per_expert": weight_bytes,
        "gradient_bytes_per_expert": gradient_bytes,
        "active_extra_copies": int(extras.sum()), "missing_active_copies": int(missing.sum()),
        "initial_active_warm_extra_copies": int(np.count_nonzero(extras & initial)),
        "missing_copies_per_rank": missing.sum(1).tolist(),
        "active_extra_copies_per_rank": extras.sum(1).tolist(),
        "forward": forward, "gradient": gradient,
        "shape_provenance": "H inherits simulator; I=2048 follows local MoonEP bench_vs_deepep default; 3 matrices assumes full SwiGLU and is not inferred from router dump. MoonEP prefetch API copies one arbitrary [H,H'] tensor.",
        "contracts": [
            "Fresh weight version after each optimizer update; no cross-update warm-cache assumption.",
            "Only executed nonhome experts require weights; unused reserved slots cost zero.",
            "Weight source is home GPU; one RDMA payload per expert/remote node, then NVLink fanout; no source-side NVLink redistribution.",
            "Gradient partials reduce within their execution node onto the home-rail relay, then one reverse RDMA payload per expert/remote node.",
            "Initial resident copies must already hold the current weight version; their acquisition costs belong to earlier work and are not free over a cold window.",
            "No switch multicast, compression, quantization scales, optimizer-state migration, HBM contention, reduction arithmetic or finite buffering modeled.",
        ],
    }


def compose_forward_cost(token_result, weight_result, planner_us=None):
    """Compose forward + replica communication, not a full training latency.

    Main result serializes weight fill before activation dispatch and replica
    gradient return after forward. The relaxed bound allows weight/dispatch
    overlap but preserves summed service on each shared TX/RX port; it is not
    a realizable overlapped schedule. Backward expert GEMMs and backward token
    movement are omitted from both quantities and must be added by the caller.
    """
    latency = token_result["latency_us"]
    planner = latency["planner"] if planner_us is None else float(planner_us)
    if planner < 0:
        raise ValueError("planner time must be nonnegative")
    weight = weight_result["forward"]["schedule"]
    gradient = weight_result["gradient"]["schedule"]
    dispatch = token_result["dispatch"]
    keys = set(weight["ports"]) | set(dispatch["ports"])
    joint_port = max((weight["ports"].get(k, {}).get("busy_us", 0.0) +
                      dispatch["ports"].get(k, {}).get("busy_us", 0.0)
                      for k in keys), default=0.0)
    rest = latency["compute"] + latency["combine"]
    return {
        "scope": "forward plus fresh-weight/replica-gradient communication; excludes backward GEMM/token movement, optimizer and reduction arithmetic",
        "planner_us_assumed": planner,
        "weight_fill_us": weight["makespan_us"],
        "replica_gradient_return_us": gradient["makespan_us"],
        "token_forward_without_planner_us": latency["dispatch"] + rest,
        "serialized_forward_weight_gradient_us": planner + weight["makespan_us"] + latency["dispatch"] + rest + gradient["makespan_us"],
        "relaxed_weight_dispatch_joint_port_bound_us": joint_port,
        "relaxed_weight_dispatch_overlap_lower_bound_us": planner + joint_port + rest + gradient["makespan_us"],
        "overlap_bound_contract": "Only weight-fill/activation-dispatch scheduling relaxed; summed same-port chunk service retained. Later compute/combine/gradient phases use their explicit separate schedules. Not a measured or achievable overlap result.",
    }
