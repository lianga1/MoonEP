"""Compare bounded CPU planners on identical saved 128K workloads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from rail_planner import optimize, finalize
from rail_support import select_support
from same_expert import optimize_same_expert
from simulate import Config, evaluate

ROOT = Path(__file__).resolve().parents[2]


def initial_support(x, owner, home):
    ranks, experts = x.shape[0], len(home)
    counts = np.bincount((owner.astype(np.int64) * experts + x).ravel(), minlength=ranks * experts).reshape(ranks, experts)
    return (np.arange(ranks)[:, None] == home[None, :]) | (counts > 0)


def execute_case(meta, input_dir, out, methods, full_graph=False):
    input_path = input_dir / meta["npz_file"]
    with np.load(input_path) as data:
        x, seed, home = data["topk"], data["owner"], data["home_rank"]
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    seed_support = initial_support(x, seed, home)
    t0 = time.perf_counter()
    support, support_diag = select_support(x, seed, home)
    support_seconds = time.perf_counter() - t0
    base_counts = None
    rows = []
    for method in methods:
        started = time.perf_counter()
        used_support = seed_support
        if method == "seed":
            owner, diag = seed.copy(), {"scope": "saved legal MoonEP-equivalent seed"}
        elif method == "same_expert":
            owner, diag = optimize_same_expert(x, seed)
        elif method == "bounded_existing":
            owner, diag = optimize(x, seed, seed_support, rounds=1)
        elif method in ("rail_b8_r1", "rail_b8_r2", "rail_b8_unweighted"):
            used_support = support
            selected_edges = None if full_graph else [(e["a"], e["b"]) for e in support_diag["accepted_edges"]]
            owner, diag = optimize(x, seed, support, rounds=2 if method == "rail_b8_r2" else 1,
                                   pressure_weighted=method != "rail_b8_unweighted", edges=selected_edges)
        else:
            raise ValueError(method)
        optimizer_seconds = time.perf_counter() - started
        t0 = time.perf_counter()
        final = finalize(x, owner, used_support, home)
        address_seconds = time.perf_counter() - t0
        npz_path = out / f"{meta['scenario']}_{method}.npz"
        np.savez_compressed(npz_path, owner=owner, support=used_support,
                            input_npz_sha256=np.array(digest), method=np.array(method),
                            scenario=np.array(meta["scenario"]), budget=np.array(8),
                            padding=np.array(128), **final)
        copies = np.full((x.shape[0], 8), -1, np.int32)
        for rank in range(x.shape[0]):
            ids = np.flatnonzero(used_support[rank] & (home != rank))
            copies[rank, :len(ids)] = ids
        simulated = evaluate(dict(topk=x, owner=owner, home_rank=home, experts_to_copy=copies), meta, Config())
        counts = simulated["traffic"]
        if method == "seed":
            base_counts = counts
        active_new = (final["final_pair_counts_rank_expert"] > 0) & ~seed_support
        new_resident = used_support & ~seed_support
        reserved_extras = (used_support & (home[None] != np.arange(x.shape[0])[:, None])).sum(1)
        row = {
            "scenario": meta["scenario"], "method": method, "input_npz_sha256": digest,
            "output_npz": npz_path.name,
            "traffic": counts,
            "modeled_latency_us": simulated["latency_us"],
            "model_scope": "uncalibrated assumed rates and phase barriers; planner/prefetch/metadata excluded",
            "reserved_extra_per_rank": reserved_extras.tolist(),
            "new_reserved_replicas": int(new_resident.sum()),
            "new_active_replicas": int(active_new.sum()),
            "new_active_replicas_cross_node": int(np.count_nonzero(active_new & ((home[None] // 8) != (np.arange(32)[:, None] // 8)))),
            "optimizer": diag,
            "support_selection": support_diag if method.startswith("rail_b8") else None,
            "cpu_reference_seconds_not_kernel_latency": {
                "support_selection": support_seconds if method.startswith("rail_b8") else 0,
                "optimizer": optimizer_seconds, "final_address_rebuild": address_seconds,
                "total_excluding_saved_seed": optimizer_seconds + address_seconds + (support_seconds if method.startswith("rail_b8") else 0),
            },
            "padding": 128, "budget": 8,
        }
        (out / f"{meta['scenario']}_{method}.json").write_text(json.dumps(row, indent=2) + "\n")
        compact = {k: row[k] for k in ("scenario", "method", "output_npz", "input_npz_sha256", "new_reserved_replicas", "new_active_replicas", "new_active_replicas_cross_node", "cpu_reference_seconds_not_kernel_latency")}
        compact.update(rdma_copies=counts["rdma_token_copies"], nvlink_copies=counts["nvlink_token_copies"],
                       rail_max_over_mean=counts["rail_statistics"]["max_over_mean"], rail_cv=counts["rail_statistics"]["cv"],
                       max_rail_copies=max(counts["rail_rdma_token_copies"]),
                       max_nic_copies=counts["max_nic_port_token_copies"],
                       max_nvlink_port_copies=counts["max_nvlink_port_token_copies"],
                       max_padded_pairs=max(counts["padded_pair_loads_per_rank"]),
                       modeled_moe_us=simulated["latency_us"]["modeled_moe_layer"],
                       max_reserved_extra=int(reserved_extras.max()))
        if base_counts is not None:
            for field, key in (("rdma_copies", "rdma_token_copies"), ("nvlink_copies", "nvlink_token_copies"), ("max_nic_copies", "max_nic_port_token_copies")):
                compact[field + "_reduction_pct"] = 100 * (1 - compact[field] / base_counts[key]) if base_counts[key] else 0
        rows.append(compact)
        print(json.dumps({k: v for k, v in compact.items() if k not in ("input_npz_sha256", "cpu_reference_seconds_not_kernel_latency")}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/multinode_rail/scenarios")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/multinode_rail/planner_sparse")
    parser.add_argument("--full-graph", action="store_true", help="ablation: allow all resident rank edges instead of the degree-4 graph")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--methods", nargs="+", default=["seed", "same_expert", "bounded_existing", "rail_b8_r1", "rail_b8_r2"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.input / "manifest.json").read_text())
    rows = []
    sources = ["rail_planner.py", "rail_support.py", "same_expert.py", "simulate.py", "run_planner.py"]
    for meta in manifest["scenarios"]:
        if args.scenarios and meta["scenario"] not in args.scenarios:
            continue
        rows.extend(execute_case(meta, args.input, args.output, args.methods, args.full_graph))
        summary = {"scope": "CPU semantic planner optimization; no GPU planner latency or physical-network speedup measured",
                   "topology": "4 nodes x 8 GPUs; 131072 tokens total; top8; E352; B8",
                   "kernel_goal": "full 32-GPU planner <=2x baseline remains unmeasured",
                   "graph_mode": "full_graph_ablation" if args.full_graph else "degree_4_selected_edges",
                   "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in sources},
                   "rows": rows}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
