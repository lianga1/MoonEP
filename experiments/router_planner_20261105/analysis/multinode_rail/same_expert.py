"""Bounded same-expert exchanges under an existing balanced allocation.

CPU semantic reference. Each source node chooses one local/remote execution
rank pair per expert, each token proposes at most once, and fixed signature
buckets exchange equal numbers of occurrences. No replicas are created.

The NumPy reference scans small bucket tables centrally and uses deterministic
priority ordering there. A parallel kernel must replace the centralized port
admission with resource quotas and include metadata/address publication. CPU
wall time is not a prediction of GPU planner latency.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import time
import numpy as np


def _traffic(owner, nodes, gpus):
    ranks, per_rank, _ = owner.shape
    f = np.zeros((nodes, gpus, nodes), np.int64)
    nvl = np.zeros((nodes, gpus, gpus), np.int64)
    for src in range(ranks):
        node, rail = divmod(src, gpus)
        mask = np.bitwise_or.reduce(np.left_shift(np.uint64(1), owner[src].astype(np.uint64)), axis=1)
        node_mask = np.bitwise_or.reduce(np.left_shift(np.uint64(1), (owner[src] // gpus).astype(np.uint64)), axis=1)
        for dst in range(nodes):
            if dst != node:
                f[node, rail, dst] = np.count_nonzero(node_mask & np.uint64(1 << dst))
        for dst in range(ranks):
            dn, dl = divmod(dst, gpus)
            if dl != rail:
                nvl[dn, rail, dl] += np.count_nonzero(mask & np.uint64(1 << dst))
    return f, nvl


def _metrics(f, nvl):
    rail = f.sum(axis=(0, 2))
    tx, rx = f.sum(2), f.sum(0).T
    return dict(rdma_copies=int(f.sum()), nvlink_copies=int(nvl.sum()),
                rail_copies=rail.tolist(), max_rail_copies=int(rail.max()),
                rail_max_over_mean=float(rail.max() / rail.mean()) if rail.mean() else None,
                max_nic_copies=int(max(tx.max(), rx.max())))


def optimize_same_expert(topk, owner, nodes=4, gpus=8, rounds=1):
    """Return (new_owner, diagnostics); never modifies input arrays.

    Source nodes coordinate their own tokens only. Every committed match swaps
    the same expert between one local and one remote existing execution rank.
    Equal expert/rank counts imply equal support, padding and pair load.
    """
    start = time.perf_counter()
    x, seed = np.asarray(topk), np.asarray(owner)
    if x.ndim != 3 or x.shape != seed.shape or x.shape[0] != nodes * gpus:
        raise ValueError("expected equal [nodes*gpus, source_tokens, top_k] arrays")
    if x.dtype.kind not in "iu" or seed.dtype.kind not in "iu" or x.size == 0:
        raise ValueError("integer nonempty routing and owner arrays required")
    if rounds < 0 or rounds > 4:
        raise ValueError("rounds must be a bounded integer between zero and four")
    ranks, per_rank, k = x.shape
    if ranks > 64 or np.any(seed < 0) or np.any(seed >= ranks) or np.any(x < 0):
        raise ValueError("invalid IDs or unsupported >64-rank bitmask")
    experts = int(x.max()) + 1
    total = ranks * per_rank
    xx = x.reshape(total, k)
    result = seed.copy()
    global_alloc = np.bincount((seed.astype(np.int64) * experts + x).ravel(), minlength=ranks * experts)
    src_rank = np.repeat(np.arange(ranks), per_rank)
    src_node, src_rail = src_rank // gpus, src_rank % gpus
    initial_f, initial_nvl = _traffic(seed, nodes, gpus)
    history = []
    operations = dict(pair_scan_upper_bound=0, eligible_pair_candidates=0,
                      proposals=0, selected_expert_edges=0, bucket_pair_tests=0,
                      accepted_swap_pairs=0, changed_execution_pairs=0)
    for iteration in range(rounds):
        round_start = time.perf_counter()
        old = result.copy()
        oo = old.reshape(total, k)
        f, nvl = _traffic(old, nodes, gpus)
        before = _metrics(f, nvl)
        # One E x R table per source node. No destination rank-pair enumeration.
        hist = np.bincount(((src_node[:, None] * experts + xx) * ranks + oo).ravel(),
                           minlength=nodes * experts * ranks).reshape(nodes, experts, ranks)
        local_rank = np.full((nodes, experts), -1, np.int16)
        remote_rank = np.full_like(local_rank, -1)
        for sn in range(nodes):
            lo, hi = sn * gpus, (sn + 1) * gpus
            local = hist[sn, :, lo:hi]
            a = local.argmax(1) + lo
            remote = hist[sn].copy()
            remote[:, lo:hi] = 0
            b = remote.argmax(1)
            valid = (hist[sn, np.arange(experts), a] > 0) & (remote[np.arange(experts), b] > 0)
            local_rank[sn, valid], remote_rank[sn, valid] = a[valid], b[valid]
        edges = int(np.count_nonzero(local_rank >= 0))
        operations["selected_expert_edges"] += edges
        operations["pair_scan_upper_bound"] += int(x.size)
        if not edges:
            history.append(dict(round=iteration, accepted_swaps=0, selected_edges=0,
                                reason="no same-expert local/remote seed quotas", before=before, after=before,
                                cpu_wall_ms=(time.perf_counter() - round_start) * 1e3))
            break
        a = local_rank[src_node[:, None], xx]
        b = remote_rank[src_node[:, None], xx]
        eligible = (a >= 0) & ((oo == a) | (oo == b))
        direction = (oo == a).astype(np.int8)  # 1 local->remote; 0 remote->local
        destination = np.where(direction == 1, b, a)
        safe_dst = np.maximum(destination, 0)
        # Two masks distinguish one occurrence from >=2 in one linear K pass.
        # No token-local K x K comparisons or T x R count matrix is needed.
        rank_seen = np.zeros(total, np.uint64)
        rank_multiple = np.zeros(total, np.uint64)
        node_seen = np.zeros(total, np.uint64)
        node_multiple = np.zeros(total, np.uint64)
        for slot in range(k):
            rb = np.left_shift(np.uint64(1), oo[:, slot].astype(np.uint64))
            nb = np.left_shift(np.uint64(1), (oo[:, slot] // gpus).astype(np.uint64))
            rank_multiple |= rank_seen & rb
            rank_seen |= rb
            node_multiple |= node_seen & nb
            node_seen |= nb
        rank_single = (rank_multiple[:, None] & np.left_shift(np.uint64(1), oo.astype(np.uint64))) == 0
        node_single = (node_multiple[:, None] & np.left_shift(np.uint64(1), (oo // gpus).astype(np.uint64))) == 0
        new_rank_exists = (rank_seen[:, None] & np.left_shift(np.uint64(1), safe_dst.astype(np.uint64))) != 0
        new_node_exists = (node_seen[:, None] & np.left_shift(np.uint64(1), (safe_dst // gpus).astype(np.uint64))) != 0
        remove = node_single & (oo // gpus != src_node[:, None])
        add = ~new_node_exists & (safe_dst // gpus != src_node[:, None])
        drdma = add.astype(np.int8) - remove.astype(np.int8)
        dnv = ((~new_rank_exists) & (safe_dst % gpus != src_rail[:, None])).astype(np.int8)
        dnv -= (rank_single & (oo % gpus != src_rail[:, None])).astype(np.int8)
        tx, rx = f.sum(2), f.sum(0).T
        rail_bytes = f.sum(axis=(0, 2))
        denom = max(1, int(max(tx.max(), rx.max())))
        remote_node = np.maximum(b, 0) // gpus
        pressure = 1.0 + 2.0 * np.maximum(tx[src_node, src_rail][:, None], rx[remote_node, src_rail[:, None]]) / denom
        pressure += rail_bytes[src_rail][:, None] / max(1, int(rail_bytes.max()))
        score = -drdma * pressure - 0.35 * dnv
        score[~eligible] = -np.inf
        choice = score.argmax(1)
        has = eligible.any(1)
        tids = np.flatnonzero(has)
        slots = choice[tids]
        operations["eligible_pair_candidates"] += int(eligible.sum())
        operations["proposals"] += len(tids)
        # Fixed E*2*G*2*3 keys. The CPU stores only nonempty buckets, without
        # sorting token records. Node and rank signatures are exact, not hashes.
        buckets = defaultdict(list)
        expert_buckets = defaultdict(lambda: [[], []])
        for t, slot in zip(tids.tolist(), slots.tolist()):
            key = (int(src_node[t]), int(xx[t, slot]), int(direction[t, slot]),
                   int(src_rail[t]), int(abs(drdma[t, slot])), int(dnv[t, slot]))
            if key not in buckets:
                expert_buckets[key[:2]][key[2]].append(key)
            buckets[key].append((t, slot))
        cursor = defaultdict(int)
        current_f = f.copy()
        max_nic, max_rail = before["max_nic_copies"], before["max_rail_copies"]
        changed = result.reshape(total, k)
        swaps = 0
        for sn in range(nodes):
            for e in np.flatnonzero(local_rank[sn] >= 0):
                rd = int(remote_rank[sn, e] // gpus)
                reverse, forward = expert_buckets[(sn, int(e))]
                candidates = []
                for p in forward:
                    for q in reverse:
                        operations["bucket_pair_tests"] += 1
                        if p[3] == q[3] or p[4] > q[4] or p[5] + q[5] > 0:
                            continue
                        # Fixed pressures evaluate the whole match. No token
                        # contributes to more than one selected bucket.
                        wp = 1 + 2 * max(tx[sn, p[3]], rx[rd, p[3]]) / denom + rail_bytes[p[3]] / max(1, max_rail)
                        wq = 1 + 2 * max(tx[sn, q[3]], rx[rd, q[3]]) / denom + rail_bytes[q[3]] / max(1, max_rail)
                        gain = q[4] * wq - p[4] * wp - 0.35 * (p[5] + q[5])
                        if gain > 1e-12:
                            candidates.append((-gain, p, q))
                # Only the bounded signature table is ordered; no full-token sort.
                candidates.sort()
                for _, p, q in candidates:
                    take = min(len(buckets[p]) - cursor[p], len(buckets[q]) - cursor[q])
                    if p[4]:
                        take = min(take, max_nic - int(current_f[sn, p[3]].sum()),
                                   max_nic - int(current_f[:, p[3], rd].sum()),
                                   max_rail - int(current_f[:, p[3], :].sum()))
                    if take <= 0:
                        continue
                    left = buckets[p][cursor[p]:cursor[p] + take]
                    right = buckets[q][cursor[q]:cursor[q] + take]
                    for (lt, lk), (rt, rk) in zip(left, right):
                        changed[lt, lk], changed[rt, rk] = oo[rt, rk], oo[lt, lk]
                    cursor[p] += take
                    cursor[q] += take
                    current_f[sn, p[3], rd] += take * p[4]
                    current_f[sn, q[3], rd] -= take * q[4]
                    swaps += take
        after_f, after_nvl = _traffic(result, nodes, gpus)
        after = _metrics(after_f, after_nvl)
        exact = np.bincount((result.astype(np.int64) * experts + x).ravel(), minlength=ranks * experts)
        assert np.array_equal(global_alloc, exact), "same-expert rank quotas changed"
        assert np.array_equal(current_f, after_f), "bucket traffic deltas differ from actual token masks"
        safe = (after["rdma_copies"] <= before["rdma_copies"] and
                after["nvlink_copies"] <= before["nvlink_copies"] and
                after["max_nic_copies"] <= before["max_nic_copies"] and
                after["max_rail_copies"] <= before["max_rail_copies"])
        if not safe:
            result = old
            history.append(dict(round=iteration, rejected_round=True, attempted_swaps=swaps,
                                before=before, rejected_after=after,
                                cpu_wall_ms=(time.perf_counter() - round_start) * 1e3))
            break
        operations["accepted_swap_pairs"] += swaps
        operations["changed_execution_pairs"] += swaps * 2
        history.append(dict(round=iteration, selected_edges=edges, proposals=len(tids),
                            nonempty_buckets=len(buckets), accepted_swaps=swaps,
                            before=before, after=after,
                            cpu_wall_ms=(time.perf_counter() - round_start) * 1e3))
    final_f, final_nvl = _traffic(result, nodes, gpus)
    diagnostics = dict(method="same_expert_one_edge_per_source_node", scope="NumPy CPU semantic reference; no GPU latency claim",
                       rounds_requested=rounds, operations=operations, history=history,
                       before=_metrics(initial_f, initial_nvl), after=_metrics(final_f, final_nvl),
                       expert_rank_counts_preserved=True, new_replicas=0,
                       cpu_wall_ms=(time.perf_counter() - start) * 1e3,
                       complexity={"candidate_scan": "O(T*K), one linear top-k pass for seen/multiple masks and one bounded proposal pass",
                                   "expert_edge_selection": "O(nodes*E*R), no R^2 rank pairs",
                                   "proposal_bucket_bound_per_source_gpu": "2*E*2*3 counters (direction, expert, RDMA toggle, NVLink delta)",
                                   "matching": "bounded source-rail/signature table per expert, centralized reference admission",
                                   "not_implemented": "parallel quota admission, raw-slot swap, final src_info publication"})
    return result, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/multinode_rail/scenarios")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    for meta in manifest["scenarios"]:
        with np.load(args.input / meta["npz_file"]) as data:
            _, d = optimize_same_expert(data["topk"], data["owner"], rounds=args.rounds)
        print(json.dumps(dict(scenario=meta["scenario"], **d)), flush=True)


if __name__ == "__main__":
    main()
