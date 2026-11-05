"""Independent exact audit of saved multirail planner assignments.

Imports neither optimizer nor simulator. Input binding is mandatory: a result
must carry input_npz_sha256 (NPZ string scalar or same-name JSON metadata).
One integer owner per input pair proves assignment cardinality only; this does
not verify physical communication addresses, numerical MoE outputs, or speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar_string(value, name):
    value = np.asarray(value)
    require(value.size == 1 and value.dtype.kind in "US", f"{name} must be a string scalar")
    item = value.reshape(-1)[0]
    return item.decode() if isinstance(item, bytes) else str(item)


def exact_traffic(owner, nodes, gpus):
    """Bitsets independently deduplicate token->rank and token->node edges."""
    ranks, _, _ = owner.shape
    require(ranks == nodes * gpus and ranks <= 64, "bitset audit requires R=N*G<=64")
    rdma = np.zeros((nodes, gpus, nodes), dtype=np.int64)
    nvl = np.zeros((nodes, gpus, gpus), dtype=np.int64)
    for src in range(ranks):
        node, rail = divmod(src, gpus)
        bits = np.bitwise_or.reduce(np.left_shift(np.uint64(1), owner[src].astype(np.uint64)), axis=-1)
        for dst_node in range(nodes):
            node_bits = np.uint64(((1 << gpus) - 1) << (dst_node * gpus))
            if dst_node != node:
                rdma[node, rail, dst_node] = np.count_nonzero(bits & node_bits)
            for local in range(gpus):
                if local != rail:
                    rank_bit = np.uint64(1 << (dst_node * gpus + local))
                    nvl[dst_node, rail, local] += np.count_nonzero(bits & rank_bit)
    return rdma, nvl


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    return {"max": float(values.max()), "mean": mean,
            "max_over_mean": float(values.max() / mean) if mean else None,
            "cv": float(values.std() / mean) if mean else None}


def allocations(topk, owner, experts):
    return np.bincount((owner.astype(np.int64) * experts + topk).ravel(),
                       minlength=topk.shape[0] * experts).reshape(topk.shape[0], experts)


def audit_physical_layout(topk, owner, home_rank, support, result, *, budget=8, padding=128):
    """Independently derive home-first physical segments and stable row indices."""
    fields = ("physical_topk", "destination_row", "physical_pair_counts", "physical_cu_seqlens")
    present = [name in result for name in fields]
    if not any(present):
        return {"physical_layout_present": False, "physical_layout_verified": False}
    require(all(present), "physical layout must include topk, rows, counts and cu_seqlens")
    ranks, _, _ = topk.shape
    experts = len(home_rank)
    home_sizes = np.bincount(home_rank, minlength=ranks)
    require(np.all(home_sizes == home_sizes[0]), "physical schema requires equal home counts")
    width = int(home_sizes[0]) + budget
    lut = np.full((ranks, experts), -1, dtype=np.int64)
    for rank in range(ranks):
        homes = np.flatnonzero(home_rank == rank)
        copies = np.flatnonzero(support[rank] & (home_rank != rank))
        ids = np.concatenate((homes, copies))
        require(len(ids) <= width, "physical slot overflow")
        lut[rank, ids] = rank * width + np.arange(len(ids))
    expected_physical = lut[owner, topk]
    physical = np.asarray(result["physical_topk"])
    require(physical.dtype.kind in "iu" and physical.shape == topk.shape, "invalid physical_topk shape or dtype")
    require(np.all(expected_physical >= 0), "physical mapping lacks a resident expert")
    require(np.array_equal(physical, expected_physical), "physical_topk does not match home-first sorted slot layout")
    counts = np.bincount(expected_physical.ravel(), minlength=ranks * width).reshape(ranks, width)
    padded = ((counts + padding - 1) // padding) * padding
    cu = np.concatenate((np.zeros((ranks, 1), dtype=np.int64), padded.cumsum(axis=1)), axis=1)
    require(np.array_equal(result["physical_pair_counts"], counts), "physical_pair_counts mismatch")
    require(np.array_equal(result["physical_cu_seqlens"], cu), "physical_cu_seqlens is not physical-slot padding prefix")
    rows = np.asarray(result["destination_row"])
    require(rows.shape == topk.shape and rows.dtype.kind in "iu", "invalid destination_row shape or dtype")
    require(np.all(rows >= 0), "negative destination row")
    start_for_pair = cu[:, :-1].ravel()[expected_physical]
    count_for_pair = counts.ravel()[expected_physical]
    require(np.all((rows >= start_for_pair) & (rows < start_for_pair + count_for_pair)),
            "destination row is outside its physical expert's unpadded interval")
    order = np.argsort(expected_physical.ravel(), kind="stable")
    group_start = np.repeat(np.cumsum(counts.ravel()) - counts.ravel(), counts.ravel())
    occurrence = np.arange(topk.size, dtype=np.int64) - group_start
    expected_rows = np.empty(topk.size, dtype=np.int64)
    expected_rows[order] = start_for_pair.ravel()[order] + occurrence
    require(np.array_equal(rows.ravel(), expected_rows), "destination rows violate stable source/token/slot ordering")
    for rank in range(ranks):
        local_rows = rows[owner == rank]
        require(len(np.unique(local_rows)) == len(local_rows), "duplicate destination row on rank")
    return {"physical_layout_present": True, "physical_layout_verified": True,
            "physical_slots_per_rank": width, "stable_pair_rows": True,
            "physical_expert_segment_membership": True, "destination_rows_unique_per_rank": True,
            "physical_padded_rows_per_rank": cu[:, -1].tolist()}


def audit_arrays(topk, input_owner, home_rank, owner, support, *, nodes=4,
                 gpus=8, slots=8, same_expert=False, padding=128,
                 saved_counts=None, saved_padded_counts=None, edge_nonincrease=False):
    topk, owner, support = map(np.asarray, (topk, owner, support))
    ranks, per_rank, k = topk.shape
    experts = len(home_rank)
    require(owner.shape == topk.shape, "owner shape does not give one assignment per input pair")
    require(owner.dtype.kind in "iu", "owner must have integer dtype")
    require(np.all((owner >= 0) & (owner < ranks)), "owner outside rank range")
    require(ranks == nodes * gpus, "input topology mismatch")
    require(np.all((topk >= 0) & (topk < experts)), "input expert outside range")
    require(np.all(np.diff(np.sort(topk, axis=-1), axis=-1) > 0), "input contains repeated expert in token")
    require(support.shape == (ranks, experts) and support.dtype.kind == "b", "support must be boolean [rank,expert]")
    require(np.all((home_rank >= 0) & (home_rank < ranks)), "invalid input home rank")
    home = np.arange(ranks)[:, None] == home_rank[None, :]
    require(np.all(support[home]), "support dropped a home expert")
    extras = (support & ~home).sum(axis=1)
    require(np.all(extras <= slots), "reserved expert slots exceed budget")
    require(np.all(support[owner, topk]), "assigned pair has no resident expert")
    counts = allocations(topk, owner, experts)
    seed_counts = allocations(topk, input_owner, experts)
    loads = counts.sum(axis=1)
    require(int(loads.sum()) == topk.size, "pair total is not conserved")
    require(np.array_equal(loads, np.full(ranks, per_rank * k)), "rank compute quotas changed")
    require(np.array_equal(counts.sum(axis=0), seed_counts.sum(axis=0)), "logical expert demand changed")
    preserved = np.array_equal(counts, seed_counts)
    if same_expert:
        require(preserved, "same-expert method changed (rank,expert) quotas")
    require(padding > 0, "padding must be positive")
    padded = ((counts + padding - 1) // padding) * padding
    if saved_counts is not None:
        require(np.array_equal(saved_counts, counts), "saved expert/rank counts disagree")
    if saved_padded_counts is not None:
        require(np.array_equal(saved_padded_counts, padded), "saved padded counts disagree")
    traffic, nvl = exact_traffic(owner, nodes, gpus)
    seed_traffic, seed_nvl = exact_traffic(input_owner, nodes, gpus)
    rail, seed_rail = traffic.sum(axis=(0, 2)), seed_traffic.sum(axis=(0, 2))
    tx, rx = traffic.sum(axis=2), traffic.sum(axis=0).T
    seed_tx, seed_rx = seed_traffic.sum(axis=2), seed_traffic.sum(axis=0).T
    if edge_nonincrease:
        require(np.all(traffic <= seed_traffic), "generic planner increased a directed RDMA edge")
        require(np.all(nvl <= seed_nvl), "generic planner increased a directed NVLink edge")
    if same_expert:
        require(traffic.sum() <= seed_traffic.sum(), "same-expert planner increased total RDMA")
        require(nvl.sum() <= seed_nvl.sum(), "same-expert planner increased total NVLink")
        require(max(tx.max(), rx.max()) <= max(seed_tx.max(), seed_rx.max()), "same-expert planner increased maximum NIC port")
        require(max(nvl.sum(axis=1).max(), nvl.sum(axis=2).max()) <= max(seed_nvl.sum(axis=1).max(), seed_nvl.sum(axis=2).max()),
                "same-expert planner increased maximum NVLink port")
    active_extras = ((counts > 0) & ~home).sum(axis=1)
    used_seed_support = home | (seed_counts > 0)
    reserved_new = support & ~used_seed_support
    return {
        "pass": True, "tokens": ranks * per_rank, "pairs": int(topk.size),
        "expected_pairs_per_rank": per_rank * k, "pair_loads_per_rank": loads.tolist(),
        "reserved_extra_experts_per_rank": extras.tolist(),
        "active_extra_experts_per_rank": active_extras.tolist(),
        "new_reserved_experts_vs_input_active_support": reserved_new.sum(axis=1).tolist(),
        "changed_pairs": int(np.count_nonzero(owner != input_owner)),
        "expert_rank_counts_preserved_vs_input": bool(preserved),
        "all_directed_rdma_edges_nonincreasing": bool(np.all(traffic <= seed_traffic)),
        "all_directed_nvlink_edges_nonincreasing": bool(np.all(nvl <= seed_nvl)),
        "expert_rank_counts_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
        "padded_pair_loads_per_rank": padded.sum(axis=1).tolist(),
        "padding_overhead_pairs": int(padded.sum() - counts.sum()),
        "rdma_copies_src_node_rail_dst_node": traffic.tolist(),
        "nvlink_copies_node_src_local_dst_local": nvl.tolist(),
        "rail_token_copies": rail.tolist(), "rail_statistics": distribution(rail),
        "nic_tx_token_copies_node_rail": tx.tolist(), "nic_rx_token_copies_node_rail": rx.tolist(),
        "rdma_token_copies": int(traffic.sum()), "nvlink_token_copies": int(nvl.sum()),
        "max_nic_port_token_copies": int(max(tx.max(), rx.max())),
        "max_nvlink_port_token_copies": int(max(nvl.sum(axis=1).max(), nvl.sum(axis=2).max())),
        "baseline": {"rdma_token_copies": int(seed_traffic.sum()), "nvlink_token_copies": int(seed_nvl.sum()),
                     "rail_statistics": distribution(seed_rail),
                     "max_nic_port_token_copies": int(max(seed_tx.max(), seed_rx.max()))},
        "checks": {"one_owner_per_pair": True, "resident_execution": True,
                   "home_support_preserved": True, "exact_rank_quotas": True,
                   "slot_budget": True, "logical_expert_demand": True,
                   "source_rail_fixed_by_construction": True},
    }


def audit_file(result_path, scenario_dir, scenario_names, *, slots=8, padding=128):
    result_path, scenario_dir = Path(result_path), Path(scenario_dir)
    sidecar = result_path.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    with np.load(result_path, allow_pickle=False) as result:
        for name in ("scenario", "method", "input_npz_sha256", "input_sha256"):
            if name in result:
                meta[name] = scalar_string(result[name], name)
        for name in ("budget", "padding"):
            if name in result:
                value = np.asarray(result[name])
                require(value.size == 1 and value.dtype.kind in "iu", f"{name} must be an integer scalar")
                meta[name] = int(value.reshape(-1)[0])
        scenario = meta.get("scenario")
        if scenario is None:
            scenario = next((s for s in sorted(scenario_names, key=len, reverse=True)
                             if result_path.stem.startswith(s + "_")), None)
        require(scenario in scenario_names, "cannot bind result to a known scenario")
        method = meta.get("method", result_path.stem[len(scenario) + 1:])
        budget = int(meta.get("budget", slots))
        effective_padding = int(meta.get("padding", meta.get("expert_padding", padding)))
        require(0 <= budget <= slots, "declared physical budget exceeds audit cap")
        input_path = scenario_dir / f"{scenario}.npz"
        actual_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        declared_hash = meta.get("input_npz_sha256", meta.get("input_sha256"))
        require(declared_hash == actual_hash, "missing or incorrect input_npz_sha256")
        scenario_meta = json.loads(input_path.with_suffix(".json").read_text())
        topology = scenario_meta["topology"]
        with np.load(input_path, allow_pickle=False) as source:
            if "topk" in result:
                require(np.array_equal(result["topk"], source["topk"]), "saved top-k changed")
            if "seed_owner" in result:
                require(np.array_equal(result["seed_owner"], source["owner"]), "saved seed owner differs from bound scenario")
            saved_counts = result["final_pair_counts_rank_expert"] if "final_pair_counts_rank_expert" in result else None
            if saved_counts is None and "pair_counts_rank_expert" in result:
                saved_counts = result["pair_counts_rank_expert"]
            saved_padded = result["padded_pair_counts_rank_expert"] if "padded_pair_counts_rank_expert" in result else None
            audited = audit_arrays(source["topk"], source["owner"], source["home_rank"],
                                   result["owner"], result["support"],
                                   nodes=topology["nodes"], gpus=topology["gpus_per_node"],
                                   slots=budget, padding=effective_padding,
                                   same_expert="same_expert" in method,
                                   edge_nonincrease=method.startswith("rail_b8_") or method.startswith("bounded_existing"),
                                   saved_counts=saved_counts, saved_padded_counts=saved_padded)
            audited["address_checks"] = audit_physical_layout(source["topk"], result["owner"], source["home_rank"],
                                                               result["support"], result, budget=budget,
                                                               padding=effective_padding)
            if "traffic" in meta:
                for key in ("rdma_copies_src_node_rail_dst_node", "nvlink_copies_node_src_local_dst_local",
                            "rdma_token_copies", "nvlink_token_copies", "max_nic_port_token_copies"):
                    if key in meta["traffic"]:
                        require(np.array_equal(meta["traffic"][key], audited[key]), f"reported traffic differs from independent count: {key}")
    audited.update(scenario=scenario, method=method, result_file=result_path.name,
                   input_npz_sha256=actual_hash,
                   result_npz_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                   input_binding_verified=True)
    return audited


def self_test():
    # A balanced 4-rank home-only assignment; cross-node relay path is present.
    x = np.array([[[2, 3]], [[0, 1]], [[0, 1]], [[2, 3]]], dtype=np.int32)
    home = np.arange(4, dtype=np.int16)
    owner = home[x]
    support = np.eye(4, dtype=bool)
    audit_arrays(x, owner, home, owner, support, nodes=2, gpus=2, slots=1)
    broken = []
    def rejected(label, changed_owner=owner, changed_support=support, same=False):
        try:
            audit_arrays(x, owner, home, changed_owner, changed_support,
                         nodes=2, gpus=2, slots=1, same_expert=same)
        except ValueError:
            broken.append(label)
        else:
            raise AssertionError(f"bad case was accepted: {label}")
    invalid = owner.copy(); invalid[0, 0, 0] = -1
    rejected("negative owner", invalid)
    unbalanced = owner.copy(); unbalanced[0, 0, 0] = 3
    rejected("nonresident or unbalanced owner", unbalanced)
    dropped = support.copy(); dropped[0, 0] = False
    rejected("dropped home", changed_support=dropped)
    rejected("slot overflow", changed_support=np.ones_like(support))
    swap = owner.copy(); swap[0, 0, 0], swap[1, 0, 0] = owner[1, 0, 0], owner[0, 0, 0]
    swap_support = support.copy(); swap_support[0, 2] = True; swap_support[2, 0] = True
    audit_arrays(x, owner, home, swap, swap_support, nodes=2, gpus=2, slots=1)
    rejected("changed same-expert quotas", swap, swap_support, True)
    rdma, nvl = exact_traffic(owner, 2, 2)
    require(int(rdma.sum()) == 2 and int(nvl.sum()) == 4, "tiny traffic reference mismatch")
    return {"valid_fixture": True, "valid_generic_swap": True,
            "invalid_cases_rejected": broken, "tiny_traffic_reference": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/multinode_rail/planner")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "outputs/multinode_rail/scenarios")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--padding", type=int, default=128)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    tests = self_test()
    if args.self_test_only:
        print(json.dumps(tests, indent=2))
        return
    manifest = json.loads((args.scenarios / "manifest.json").read_text())
    names = [row["scenario"] for row in manifest["scenarios"]]
    files = sorted(args.input.glob("*.npz"))
    require(bool(files), "no saved planner results found; empty audit is not a pass")
    rows, failures = [], []
    for path in files:
        try:
            row = audit_file(path, args.scenarios, names, slots=args.slots, padding=args.padding)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("scenario", "method", "pass", "rdma_token_copies", "nvlink_token_copies", "max_nic_port_token_copies")}))
        except (ValueError, KeyError, OSError) as exc:
            failure = {"result_file": path.name, "pass": False, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(json.dumps(failure))
    output = args.output or args.input / "audit.json"
    report = {"scope": "Independent assignment, physical-slot/row-layout and traffic audit; no actual remote-store, numerical output or hardware timing validation",
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "all_pass": not failures, "artifact_count": len(files),
              "self_tests": tests, "rows": rows, "failures": failures}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
