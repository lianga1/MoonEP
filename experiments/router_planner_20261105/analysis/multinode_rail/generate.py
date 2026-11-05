"""Reproducible 4-node, 8-rail EP workloads; CPU semantics, never GPU timing.

The pressure recipes are deliberately constructed, not fitted production data.
Node-translation symmetry and an eight-token local-rank rotation give exact
per-rank execution-pair balance without hiding communication imbalance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 20261102


def _ratio(values):
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    return {
        "max_over_mean": float(a.max() / mean) if mean else None,
        "cv": float(a.std() / mean) if mean else None,
        "max_over_min": float(a.max() / a.min()) if a.min() else None,
        "mean": mean,
        "std_population": float(a.std()),
    }


def summarize(topk, owner, home_rank, nodes=4, gpus=8):
    """Count token->node RDMA copies, keeping source rail invariant."""
    ranks, per_rank, k = topk.shape
    if ranks != nodes * gpus or owner.shape != topk.shape:
        raise ValueError("incompatible topology or owner shape")
    if np.any(owner < 0) or np.any(owner >= ranks):
        raise ValueError("invalid execution rank")
    traffic = np.zeros((nodes, gpus, nodes), dtype=np.int64)
    remote_pairs = np.zeros((nodes, gpus), dtype=np.int64)
    rank_traffic = np.zeros((ranks, ranks), dtype=np.int64)
    nvlink_copies = 0
    node_visits = np.zeros(nodes, dtype=np.int64)
    counts = np.zeros((ranks, len(home_rank)), dtype=np.int64)
    for src in range(ranks):
        n, rail = divmod(src, gpus)
        dst_node = owner[src] // gpus
        remote_pairs[n, rail] = np.count_nonzero(dst_node != n)
        for d in range(nodes):
            if d != n:
                traffic[n, rail, d] = np.count_nonzero(np.any(dst_node == d, axis=-1))
        node_visits += np.bincount(
            np.stack([np.any(dst_node == d, axis=-1) for d in range(nodes) if d != n], axis=-1).sum(-1),
            minlength=nodes,
        )
        for d in range(ranks):
            rank_traffic[src, d] = np.count_nonzero(np.any(owner[src] == d, axis=-1))
            # Remote-node forwarding starts at the same local rank as src;
            # local-node delivery starts at src itself. Both need one NVLink
            # copy per distinct destination rank with a different local rank.
            if d % gpus != rail:
                nvlink_copies += int(rank_traffic[src, d])
        np.add.at(counts, (owner[src].ravel(), topk[src].ravel()), 1)
    loads = counts.sum(-1)
    tx = traffic.sum(-1)
    rx = traffic.sum(0).T
    rail = traffic.sum(axis=(0, 2))
    extras = ((counts > 0) & (np.arange(ranks)[:, None] != home_rank[None, :])).sum(-1)
    unique_rows = bool(np.all(np.diff(np.sort(topk, axis=-1), axis=-1) > 0))
    return {
        "tokens": int(ranks * per_rank), "top_k": k,
        "shape": list(topk.shape), "pairs": int(topk.size),
        "pair_loads_per_rank": loads.tolist(),
        "pair_loads_per_node": loads.reshape(nodes, gpus).sum(-1).tolist(),
        "pair_load_max_over_mean": float(loads.max() / loads.mean()),
        "exact_equal_rank_pair_load": bool(np.all(loads == loads[0])),
        "rail_rdma_token_copies": rail.tolist(),
        "rail_statistics": _ratio(rail),
        "rdma_token_copies": int(rail.sum()),
        "remote_pairs": int(remote_pairs.sum()),
        "remote_pairs_per_source_node_rail": remote_pairs.tolist(),
        "rdma_copies_per_source_node_rail_destination_node": traffic.tolist(),
        "nic_tx_token_copies": tx.tolist(), "nic_rx_token_copies": rx.tolist(),
        "remote_node_fanout_histogram": node_visits.tolist(),
        "rank_destination_copies_including_source": int(rank_traffic.sum()),
        "nvlink_token_copies": nvlink_copies,
        "active_extra_experts_per_rank": extras.tolist(),
        "unique_experts_per_token": unique_rows,
        "observed_experts": int(np.count_nonzero(counts.sum(0))),
        "expert_count_max_over_mean": float(counts.sum(0).max() / (topk.size / len(home_rank))),
        "expert_pair_counts_sha256": hashlib.sha256(counts.sum(0).astype("<i8").tobytes()).hexdigest(),
    }


def _spread_integer(total, parts):
    """Near-uniform integer allocation whose sum is exact."""
    return np.diff(np.rint(np.linspace(0, total, parts + 1)).astype(np.int64))


def generate_synthetic(scenario="fanout_skew_2p0", *, nodes=4, gpus=8,
                       tokens=131072, experts=352, seed=DEFAULT_SEED):
    """Return (arrays, metadata) for a legal exactly compute-balanced recipe.

    Nodes/ranks are currently fixed to the user's 4x8 topology. The token count
    can be changed, provided every source rank contains whole eight-token blocks.
    Integer rounding may slightly perturb requested ratios at tiny token counts.
    """
    if (nodes, gpus) != (4, 8):
        raise ValueError("these calibrated recipes require nodes=4, gpus=8")
    ranks = nodes * gpus
    if tokens <= 0 or tokens % (ranks * gpus) or experts % ranks:
        raise ValueError("tokens must divide into rank-local 8-token blocks; E must divide R")
    per_rank, k = tokens // ranks, 8
    blocks, local_experts = per_rank // gpus, experts // ranks
    if local_experts < k:
        raise ValueError("this unique-offset recipe needs >=8 home experts per rank")
    rng = np.random.default_rng(seed)
    # Each value is the number of distinct remote destination nodes for a block.
    fanout = np.ones((gpus, blocks), dtype=np.int16)
    kind = "fanout"
    requested_ratio = 1.0
    if scenario in ("balanced", "balanced_fanout2"):
        extra = blocks // 2 if scenario == "balanced" else blocks
        for rail in range(gpus):
            fanout[rail, :extra] = 2
    elif scenario in ("fanout_skew_1p5", "fanout_skew_2p0"):
        requested_ratio = 1.5 if scenario.endswith("1p5") else 2.0
        # Both pressure cases have the same group mean fanout (1.5) as
        # balanced; only its distribution over rails changes.
        if requested_ratio == 1.5:
            fanout[0] = 2
            fanout[0, :round(blocks / 4)] = 3
        else:
            fanout[0] = 3
        cold_total = round(gpus * blocks * 1.5 - fanout[0].sum())
        increments = _spread_integer(cold_total - (gpus - 1) * blocks, gpus - 1)
        if np.any(increments < 0) or np.any(increments > blocks):
            raise ValueError("fanout recipe cannot realize requested ratio")
        for rail, extra in enumerate(increments, start=1):
            fanout[rail, :extra] = 2
    elif scenario == "locality_skew_2p0":
        kind, requested_ratio = "locality", 2.0
        fanout[:] = 0
        fanout[0] = 1
        # Hot rail has one copy/token; group average is 0.5 copy/token.
        cold_counts = _spread_integer(round(gpus * blocks / requested_ratio - blocks), gpus - 1)
        for rail, count in enumerate(cold_counts, start=1):
            fanout[rail, :count] = 1
    else:
        raise ValueError(f"unknown synthetic scenario: {scenario}")
    for row in fanout:
        rng.shuffle(row)

    topk = np.empty((ranks, per_rank, k), dtype=np.int32)
    owner = np.empty_like(topk, dtype=np.int16)
    phases = np.arange(gpus)[:, None]
    for rail in range(gpus):
        for block in range(blocks):
            q = int(fanout[rail, block])
            dest_offsets = np.zeros(k, dtype=np.int16)
            if q:
                remote_choices = rng.permutation(np.arange(1, nodes))[:q]
                remote_count = 4 if kind == "fanout" else k
                dest_offsets[:remote_count] = np.resize(remote_choices, remote_count)
                rng.shuffle(dest_offsets)
            # A permutation fixes execution-rank fanout at eight and NVLink
            # forwarding at seven copies/token in every synthetic scenario.
            # Eight phases make every slot hit every local destination rank once.
            local_shifts = rng.permutation(gpus)
            local_dst = (phases + local_shifts[None, :]) % gpus
            # Every token has distinct home-expert offsets, even if all K pairs
            # happen to land on one rank. Sample a fresh set for each token.
            expert_offsets = np.stack([rng.permutation(local_experts)[:k] for _ in range(gpus)])
            sl = slice(block * gpus, (block + 1) * gpus)
            for source_node in range(nodes):
                rank = source_node * gpus + rail
                dst = ((source_node + dest_offsets[None, :]) % nodes) * gpus + local_dst
                owner[rank, sl] = dst
                topk[rank, sl] = dst * local_experts + expert_offsets

    home_rank = np.arange(experts, dtype=np.int16) // local_experts
    assert np.array_equal(home_rank[topk], owner)
    stats = summarize(topk, owner, home_rank, nodes, gpus)
    assert stats["exact_equal_rank_pair_load"] and stats["unique_experts_per_token"]
    assert stats["pair_loads_per_rank"] == [per_rank * k] * ranks
    if kind == "fanout":
        assert stats["remote_pairs"] == tokens * 4
    arrays = {"topk": topk, "owner": owner, "home_rank": home_rank,
              "block_remote_node_fanout": fanout}
    metadata = {
        "scenario": scenario, "provenance": "constructed_synthetic_pressure_workload",
        "seed": seed, "topology": {"nodes": nodes, "gpus_per_node": gpus, "rails": gpus},
        "expert_count": experts, "requested_rail_max_over_mean": requested_ratio,
        "owner_semantics": "home_rank[topk]; zero replicas; exact per-rank compute balance",
        "construction": "node-translation symmetry; block-local destination-rank rotation; randomized legal home-expert subsets",
        "mechanism": kind,
        "synthetic_limitations": [
            "No measured production multi-node distribution is available; target ratio is a stress constraint, not a statistical fit.",
            "Node-symmetric demand and 8-token blocks are deliberate; temporal correlation and arbitrary per-port incast are not reproduced.",
            "Expert popularity and cooccurrence are randomized within the construction, not matched to the real router dump.",
            "Each synthetic token visits every local-rank index exactly once: execution-rank fanout is fixed at 8, and NVLink copies are fixed at 7/token. Trace-lift fanout remains natural.",
            "No gate weights, token payload, sequence boundaries, or real timing are supplied.",
        ],
        "statistics": stats,
    }
    return arrays, metadata


def generate_trace_lift(layer, *, tokens=131072, seed=DEFAULT_SEED, budget=8):
    """Preserve complete real top-k rows, but invent node labels and placement."""
    sys.path.insert(0, str(ROOT / "analysis"))
    from moonep_cpu import plan

    path = ROOT / f"outputs/trace_audit/layer_{layer:02d}.npz"
    raw = np.load(path)["topk"]
    nodes, gpus, ranks = 4, 8, 32
    if tokens % ranks:
        raise ValueError("trace lift requires equal integer tokens per new rank")
    per_rank = tokens // ranks
    per_original_rank = per_rank * nodes
    if raw.shape[0] != gpus or per_original_rank > raw.shape[1]:
        raise ValueError("not enough source rows for without-replacement lift")
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.choice(raw.shape[1], per_original_rank, replace=False) for _ in range(gpus)])
    sampled = raw[np.arange(gpus)[:, None], indices]
    topk = sampled.reshape(gpus, nodes, per_rank, raw.shape[-1]).transpose(1, 0, 2, 3).reshape(ranks, per_rank, raw.shape[-1]).astype(np.int32)
    original_indices = indices.reshape(gpus, nodes, per_rank).transpose(1, 0, 2).reshape(ranks, per_rank)
    home_rank = np.arange(352, dtype=np.int16) // 11
    arrays = {"topk": topk, "home_rank": home_rank,
              "original_source_rank": np.arange(ranks, dtype=np.int16) % gpus,
              "original_source_token_index": original_indices.astype(np.int32)}
    metadata = {
        "scenario": f"trace_lift_layer{layer:02d}", "provenance": "real_topk_rows_with_synthetic_multinode_mapping",
        "seed": seed, "source_path": str(path.relative_to(ROOT)),
        "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_shape": list(raw.shape), "sampling": "without replacement per original source rank; complete top-k rows preserved",
        "topology": {"nodes": nodes, "gpus_per_node": gpus, "rails": gpus},
        "expert_count": 352, "budget_per_rank": budget,
        "original_source_labels": "original rank mapped to local rank; sampled rows partitioned across 4 invented nodes",
        "home_mapping": "synthetic contiguous home_rank[e]=e//11, not a measured production placement",
        "synthetic_limitations": [
            "Not a real multi-node trace; node labels and 32-rank home placement are invented.",
            "One original iteration/microbatch only; no evidence of cross-step persistence.",
            "No gate weights or valid-token mask in the original dump.",
            "The rail ratio is measured after lifting; it is not forced into the 1.5-2 target interval.",
        ],
    }
    try:
        p = plan(topk, E=352, B=budget, replica_target=0)
        arrays.update(owner=p.owner.astype(np.int16), experts_to_copy=p.experts_to_copy)
        stats = summarize(topk, p.owner, home_rank, nodes, gpus)
        invalid = int(np.count_nonzero(p.physical_topk < 0))
        metadata.update(statistics=stats, unsupported_pairs=invalid,
                        legal_plan=bool(invalid == 0 and max(stats["active_extra_experts_per_rank"]) <= budget),
                        owner_semantics="existing CPU MoonEP semantic planner, rt0; not a new 32-GPU kernel benchmark")
    except Exception as exc:
        metadata.update(legal_plan=False, planner_error=f"{type(exc).__name__}: {exc}")
    return arrays, metadata


def write_scenario(out, arrays, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = metadata["scenario"]
    path = out / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    metadata["npz_file"] = path.name
    metadata["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["array_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()}
    (out / f"{name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/multinode_rail/scenarios")
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-trace", action="store_true")
    args = parser.parse_args()
    rows = []
    for name in ("balanced", "balanced_fanout2", "fanout_skew_1p5", "fanout_skew_2p0", "locality_skew_2p0"):
        arrays, meta = generate_synthetic(name, tokens=args.tokens, seed=args.seed)
        rows.append(write_scenario(args.out, arrays, meta))
        print(name, meta["statistics"]["rail_statistics"], flush=True)
    if not args.skip_trace:
        for layer in (0, 7):
            arrays, meta = generate_trace_lift(layer, tokens=args.tokens, seed=args.seed + layer)
            rows.append(write_scenario(args.out, arrays, meta))
            print(meta["scenario"], {"legal_plan": meta["legal_plan"], "unsupported_pairs": meta.get("unsupported_pairs"),
                "rail": meta.get("statistics", {}).get("rail_statistics")}, flush=True)
    manifest = {"schema_version": 1, "scope": "CPU workload generation and exact traffic counts; no physical-network or GPU timing",
                "generator": "analysis/multinode_rail/generate.py", "seed": args.seed,
                "token_scope": "global EP group", "tokens": args.tokens, "scenarios": rows}
    matched = [r for r in rows if r["scenario"] in ("balanced", "fanout_skew_1p5", "fanout_skew_2p0")]
    manifest["controlled_comparison"] = {
        "scenarios": [r["scenario"] for r in matched],
        "same_global_expert_pair_histogram": len({r["statistics"]["expert_pair_counts_sha256"] for r in matched}) == 1,
        "same_total_rdma_copies": len({r["statistics"]["rdma_token_copies"] for r in matched}) == 1,
        "same_total_remote_pairs": len({r["statistics"]["remote_pairs"] for r in matched}) == 1,
        "same_exact_per_rank_pair_loads": len({tuple(r["statistics"]["pair_loads_per_rank"]) for r in matched}) == 1,
        "same_total_nvlink_copies": len({r["statistics"]["nvlink_token_copies"] for r in matched}) == 1,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
