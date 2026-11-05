"""Independent semantic checks for rail workloads and a same-expert swap.

No generator/simulator counters are imported. These checks establish counting
and allocation invariants, not numerical MoE-output equivalence or GPU speed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def independent_counts(owner, nodes, gpus):
    ranks, tokens, k = owner.shape
    assert ranks == nodes * gpus and np.all((owner >= 0) & (owner < ranks))
    rdma = np.zeros((nodes, gpus, nodes), dtype=np.int64)
    nvlink = np.zeros((nodes, gpus, gpus), dtype=np.int64)
    for src in range(ranks):
        src_node, rail = divmod(src, gpus)
        rank_mask = np.bitwise_or.reduce(np.left_shift(np.uint64(1), owner[src].astype(np.uint64)), axis=1)
        node_mask = np.bitwise_or.reduce(np.left_shift(np.uint64(1), (owner[src] // gpus).astype(np.uint64)), axis=1)
        for dst_node in range(nodes):
            if dst_node != src_node:
                rdma[src_node, rail, dst_node] = np.count_nonzero(node_mask & np.uint64(1 << dst_node))
        for dst in range(ranks):
            dst_node, local = divmod(dst, gpus)
            if local != rail:
                nvlink[dst_node, rail, local] += np.count_nonzero(rank_mask & np.uint64(1 << dst))
    return rdma, nvlink


def unit_checks():
    owner = np.broadcast_to(np.arange(4, dtype=np.int16)[:, None, None], (4, 1, 2)).copy()
    owner[0, 0] = [2, 3]
    rdma, nvl = independent_counts(owner, 2, 2)
    assert rdma.sum() == 1 and rdma[0, 0, 1] == 1
    assert nvl.sum() == 1 and nvl[1, 0, 1] == 1
    permuted = (owner // 2) * 2 + (1 - owner % 2)
    assert np.array_equal(rdma, independent_counts(permuted, 2, 2)[0])
    assert independent_counts(owner % 2, 1, 4)[0].sum() == 0

    # Two same-source-node tokens execute expert 0 at ranks 3 and 1.
    # Token B already visits the remote node for expert 2; swapping expert 0
    # removes A's hot-rail transfer without adding a new B transfer.
    topk = np.array([[[0, 1]], [[0, 2]], [[3, 4]], [[5, 6]]], dtype=np.int32)
    before = np.array([[[3, 0]], [[1, 2]], [[2, 2]], [[3, 3]]], dtype=np.int16)
    after = before.copy()
    after[0, 0, 0], after[1, 0, 0] = before[1, 0, 0], before[0, 0, 0]
    def alloc(a):
        return np.bincount((a * 7 + topk).ravel(), minlength=28).reshape(4, 7)
    assert np.array_equal(alloc(before), alloc(after))
    b_rdma, b_nvl = independent_counts(before, 2, 2)
    a_rdma, a_nvl = independent_counts(after, 2, 2)
    assert b_rdma.sum() == 2 and a_rdma.sum() == 1
    assert b_rdma[:, 0, :].sum() == 1 and a_rdma[:, 0, :].sum() == 0
    assert a_rdma[:, 1, :].sum() == b_rdma[:, 1, :].sum() == 1
    assert b_nvl.sum() == a_nvl.sum()
    return {"two_stage_path": True, "node_dedup": True,
            "destination_local_permutation_preserves_rdma": True,
            "single_node_zero_rdma": True,
            "same_expert_swap_preserves_expert_rank_counts": True,
            "swap_example_rdma_copies_before_after": [int(b_rdma.sum()), int(a_rdma.sum())],
            "swap_example_nvlink_copies_before_after": [int(b_nvl.sum()), int(a_nvl.sum())]}


def simulator_checks():
    from simulate import Config, simulate_chunks
    config = Config(hidden=16, chunk_tokens=1)
    owner = np.broadcast_to(np.arange(4, dtype=np.int16)[:, None, None], (4, 1, 2)).copy()
    owner[0, 0] = [2, 3]
    dispatch, combine, completion = simulate_chunks(owner, 2, 2, config)
    for scheduler in (dispatch, combine):
        for resource in scheduler.free:
            events = sorted((r for r in scheduler.records if resource in r["resources"]), key=lambda r: r["start_us"])
            assert all(a["end_us"] <= b["start_us"] for a, b in zip(events, events[1:]))
        assert sum(r["kind"] == "rdma" for r in scheduler.records) == 1
        assert sum(r["kind"] == "nvlink" for r in scheduler.records) == 1
        assert all(r["start_us"] >= r["ready_us"] for r in scheduler.records)
    d_rdma = next(r for r in dispatch.records if r["kind"] == "rdma")
    d_nvl = next(r for r in dispatch.records if r["kind"] == "nvlink")
    c_rdma = next(r for r in combine.records if r["kind"] == "rdma")
    c_nvl = next(r for r in combine.records if r["kind"] == "nvlink")
    assert d_nvl["start_us"] >= d_rdma["end_us"]
    assert c_rdma["start_us"] >= c_nvl["end_us"]
    assert d_rdma["source_node"] == c_rdma["destination_node"] == 0
    assert d_rdma["source_local"] == c_rdma["destination_local"] == 0
    assert completion[0, 0] == c_rdma["end_us"]
    return {"chunk_dependencies": True, "resource_exclusion": True,
            "reverse_combine": True}


def main():
    path = ROOT / "outputs/multinode_rail/scenarios"
    manifest = json.loads((path / "manifest.json").read_text())
    rows, control = [], {}
    for meta in manifest["scenarios"]:
        name = meta["scenario"]
        data = np.load(path / meta["npz_file"])
        x, owner, home = data["topk"], data["owner"], data["home_rank"]
        assert x.shape == owner.shape == (32, 4096, 8)
        assert np.all(np.diff(np.sort(x, axis=-1), axis=-1) > 0)
        assert np.all((x >= 0) & (x < len(home)))
        counts = np.bincount((owner.astype(np.int64) * len(home) + x).ravel(), minlength=32 * len(home)).reshape(32, len(home))
        assert np.array_equal(counts.sum(1), np.full(32, 32768))
        if "experts_to_copy" in data:
            support = home[None, :] == np.arange(32)[:, None]
            for rank, copies in enumerate(data["experts_to_copy"]):
                support[rank, copies[copies >= 0]] = True
            assert np.all((counts == 0) | support)
            assert np.max((support & (home[None, :] != np.arange(32)[:, None])).sum(1)) <= 8
        else:
            assert np.array_equal(home[x], owner)
        rdma, nvl = independent_counts(owner, 4, 8)
        assert np.array_equal(rdma, meta["statistics"]["rdma_copies_per_source_node_rail_destination_node"])
        simulated_path = ROOT / f"outputs/multinode_rail/simulation/{name}.json"
        if simulated_path.exists():
            simulated = json.loads(simulated_path.read_text())
            st, cfg = simulated["traffic"], simulated["config"]
            assert simulated["input_npz_sha256"] == hashlib.sha256((path / meta["npz_file"]).read_bytes()).hexdigest()
            assert simulated["simulator_sha256"] == hashlib.sha256((Path(__file__).parent / "simulate.py").read_bytes()).hexdigest()
            assert np.array_equal(rdma, st["rdma_copies_src_node_rail_dst_node"])
            assert np.array_equal(nvl, st["nvlink_copies_node_src_local_dst_local"])
            assert np.array_equal(counts, st["pair_counts_rank_expert"])
            for phase, dtype in (("dispatch", "input_bytes_per_element"), ("combine", "output_bytes_per_element")):
                assert simulated[phase]["rdma_payload_bytes"] == rdma.sum() * cfg["hidden"] * cfg[dtype]
                assert simulated[phase]["nvlink_payload_bytes"] == nvl.sum() * cfg["hidden"] * cfg[dtype]
        if "original_source_token_index" in data:
            raw = np.load(ROOT / meta["source_path"])["topk"]
            for rank in range(32):
                original_rank = int(data["original_source_rank"][rank])
                assert np.array_equal(x[rank], raw[original_rank, data["original_source_token_index"][rank]])
            for rail in range(8):
                ids = data["original_source_token_index"][rail::8].ravel()
                assert len(np.unique(ids)) == len(ids)
        rail = rdma.sum(axis=(0, 2))
        rows.append({"scenario": name, "pass": True,
                     "rdma_token_copies": int(rdma.sum()), "nvlink_token_copies": int(nvl.sum()),
                     "rail_max_over_mean": float(rail.max() / rail.mean()),
                     "input_sha256": hashlib.sha256((path / meta["npz_file"]).read_bytes()).hexdigest()})
        if name in ("balanced", "fanout_skew_1p5", "fanout_skew_2p0"):
            control[name] = (counts.sum(0), rdma.sum(), nvl.sum())
    base = control["balanced"]
    for value in control.values():
        assert np.array_equal(base[0], value[0]) and base[1:] == value[1:]
    result = {"scope": "Independent exact allocation and traffic-count invariants; no GPU or numerical MoE test",
              "unit_checks": unit_checks(), "simulator_checks": simulator_checks(),
              "controlled_comparison": True, "scenarios": rows}
    out = ROOT / "outputs/multinode_rail/independent_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
