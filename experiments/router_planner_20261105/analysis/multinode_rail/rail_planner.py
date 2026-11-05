"""Bounded-candidate CPU planner reference, not a GPU timing implementation.

Each execution pair checks at most two resident targets: the source GPU (or
an already visited GPU in the source node), and another already visited GPU.
One proposal per token, reciprocal equal-count matching, one default round.
Every move only removes payload edges: it never adds an RDMA or NVLink edge.
The proposed kernel uses integer ffs/popcount and counting-prefix primitives;
this NumPy reference also uses linear counting, with no token comparison sort.
"""
from __future__ import annotations
import time
import numpy as np


def token_masks(owner, gpus=8):
    o = owner.reshape(-1, owner.shape[-1]).astype(np.int64)
    seen = np.zeros(len(o), np.uint64)
    multi = seen.copy()
    node_seen = seen.copy()
    node_multi = seen.copy()
    for k in range(o.shape[1]):
        bit = np.left_shift(np.uint64(1), o[:, k].astype(np.uint64))
        multi |= seen & bit
        seen |= bit
        bit = np.left_shift(np.uint64(1), (o[:, k] // gpus).astype(np.uint64))
        node_multi |= node_seen & bit
        node_seen |= bit
    return seen, seen & ~multi, node_seen, node_seen & ~node_multi


def _choose_bit(mask, offsets, ranks):
    """Rotating ffs; log2 is exact on powers of two here (R<=32)."""
    offsets = np.asarray(offsets, dtype=np.uint64)
    all_bits = np.uint64((1 << ranks) - 1)
    rotated = ((mask >> offsets) | (mask << (np.uint64(ranks) - offsets))) & all_bits
    low = rotated & (~rotated + np.uint64(1))
    first = np.log2(np.maximum(low, np.uint64(1))).astype(np.int64)
    return np.where(mask != 0, (first + offsets.astype(np.int64)) % ranks, -1)


def edge_counts(owner, nodes=4, gpus=8, return_masks=False):
    """O(TK) dedup/histogram scan, independent of the total rank count."""
    ranks, per_rank, _ = owner.shape
    rdma = np.zeros((nodes, gpus, nodes), np.int64)
    nv = np.zeros((nodes, gpus, gpus), np.int64)
    o = owner.reshape(-1, owner.shape[-1]).astype(np.int64)
    src = np.repeat(np.arange(ranks), per_rank)
    sn, rail = src // gpus, src % gpus
    seen = np.zeros(len(o), np.uint64)
    multi = seen.copy()
    ns = seen.copy()
    nm = seen.copy()
    for k in range(o.shape[1]):
        dst = o[:, k]
        dn, dl = dst // gpus, dst % gpus
        rb = np.left_shift(np.uint64(1), dst.astype(np.uint64))
        nb = np.left_shift(np.uint64(1), dn.astype(np.uint64))
        new_rank = ((seen & rb) == 0) & (dl != rail)
        new_node = ((ns & nb) == 0) & (dn != sn)
        nv += np.bincount(((dn * gpus + rail) * gpus + dl)[new_rank], minlength=nodes * gpus * gpus).reshape(nv.shape)
        rdma += np.bincount(((sn * gpus + rail) * nodes + dn)[new_node], minlength=nodes * gpus * nodes).reshape(rdma.shape)
        multi |= seen & rb
        seen |= rb
        nm |= ns & nb
        ns |= nb
    if return_masks:
        return rdma, nv, (seen, seen & ~multi, ns, ns & ~nm)
    return rdma, nv


def _round(topk, owner, support, nodes, gpus, step=0, pressure_weighted=True, edges=None):
    ranks, per_rank, kmax = topk.shape
    if ranks > 32:
        raise ValueError("this prototype supports at most 32 ranks")
    x = topk.reshape(-1, kmax)
    o = owner.reshape(-1, kmax)
    tokens = len(x)
    src = np.repeat(np.arange(ranks), per_rank)
    sn, rail = src // gpus, src % gpus
    token_id = np.arange(tokens)
    flow, nv_before, masks = edge_counts(owner, nodes, gpus, return_masks=True)
    seen, singleton, node_seen, node_singleton = masks
    resident = (support.T.astype(np.uint64) << np.arange(ranks, dtype=np.uint64)).sum(axis=1, dtype=np.uint64)
    if edges is None:
        edges = [(a, b) for a in range(ranks) for b in range(a + 1, ranks)]
    else:
        edges = sorted(set(tuple(sorted((int(a), int(b)))) for a, b in edges))
    edge_lut = np.full((ranks, ranks), -1, np.int32)
    neighbors = np.zeros(ranks, np.uint64)
    degrees = np.zeros(ranks, np.int32)
    for i, (a, b) in enumerate(edges):
        if not 0 <= a < b < ranks:
            raise ValueError("invalid rank edge")
        edge_lut[a, b], edge_lut[b, a] = 2 * i, 2 * i + 1
        neighbors[a] |= np.uint64(1 << b)
        neighbors[b] |= np.uint64(1 << a)
        degrees[a] += 1
        degrees[b] += 1
    src_bit = np.left_shift(np.uint64(1), src.astype(np.uint64))
    src_node_mask = np.left_shift(np.uint64((1 << gpus) - 1), (sn * gpus).astype(np.uint64))
    rail_load = flow.sum(axis=(0, 2)).astype(float)
    ratio = rail_load / max(float(rail_load.mean()), 1.0)
    if not pressure_weighted:
        ratio[:] = 1.0
    pressure = 1 + 4 * ratio[rail] ** 4
    priority = np.clip(np.rint(2 + 3 * ratio[rail]), 2, 7).astype(np.int64)
    best_score = np.full(tokens, -np.inf)
    best_slot = np.full(tokens, -1, np.int16)
    best_target = np.full(tokens, -1, np.int16)
    best_class = np.zeros(tokens, np.int16)
    best_d = np.zeros(tokens, np.int16)
    best_v = np.zeros(tokens, np.int16)
    evaluated = resident_candidates = positive_candidates = 0
    for k in range(kmax):
        a = o[:, k].astype(np.int64)
        ab = np.left_shift(np.uint64(1), a.astype(np.uint64))
        avail = resident[x[:, k]] & ~ab & neighbors[a]
        offset = ((x[:, k].astype(np.int64) * 17 + token_id + 13 * step) % ranks).astype(np.uint64)
        local = avail & seen & src_node_mask
        first = _choose_bit(local, offset, ranks)
        first = np.where((avail & src_bit) != 0, src, first)
        first_bit = np.where(first >= 0, np.left_shift(np.uint64(1), np.maximum(first, 0).astype(np.uint64)), np.uint64(0))
        second = _choose_bit(avail & seen & ~first_bit, offset, ranks)
        for candidate in (first, second):
            evaluated += tokens
            valid = candidate >= 0
            b = np.maximum(candidate, 0)
            bb = np.left_shift(np.uint64(1), b.astype(np.uint64))
            an, bn = a // gpus, b // gpus
            an_bit = np.left_shift(np.uint64(1), an.astype(np.uint64))
            bn_bit = np.left_shift(np.uint64(1), bn.astype(np.uint64))
            removed = (an != sn) & ((node_singleton & an_bit) != 0)
            added = (bn != sn) & ((node_seen & bn_bit) == 0)
            d_gain = np.where(an != bn, removed.astype(np.int16) - added.astype(np.int16), 0)
            v_gain = (((a % gpus != rail) & ((singleton & ab) != 0)).astype(np.int16)
                      - ((b % gpus != rail) & ((seen & bb) == 0)).astype(np.int16))
            # These asserts express the restricted target-set guarantee.
            assert not np.any(valid & (added | (v_gain < 0) | (d_gain < 0)))
            resident_candidates += int(valid.sum())
            positive_candidates += int(np.count_nonzero(valid & ((d_gain > 0) | (v_gain > 0))))
            score = pressure * d_gain + 0.5 * v_gain
            # Rotating deterministic tie, not a global token sort.
            tie = ((a * 31 + b * 13 + k + token_id + step) % 127) / 100000.0
            score += tie
            improve = valid & (score > best_score)
            best_score[improve] = score[improve]
            best_slot[improve] = k
            best_target[improve] = b[improve]
            cls = np.where(d_gain > 0, priority, np.where(v_gain > 0, 1, 0))
            best_class[improve] = cls[improve]
            best_d[improve] = d_gain[improve]
            best_v[improve] = v_gain[improve]
    valid_ids = np.flatnonzero(best_slot >= 0)
    old_rank = o[valid_ids, best_slot[valid_ids]].astype(np.int64)
    keys = edge_lut[old_rank, best_target[valid_ids]] * 8 + best_class[valid_ids]
    # Exact stable counting ordinals, O(T + edges*Q), deliberately no argsort.
    counts = np.zeros(2 * len(edges) * 8, np.int64)
    ordinal = np.empty(len(valid_ids), np.int64)
    for i, key in enumerate(keys):
        ordinal[i] = counts[key]
        counts[key] += 1
    counts = counts.reshape(2 * len(edges), 8)
    accept = np.zeros_like(counts)
    paired = 0
    for i in range(len(edges)):
        ab, ba = 2 * i, 2 * i + 1
        # At least one side of every conceptual pair has positive gain.
        n = min(int(counts[ab].sum()), int(counts[ba].sum()),
                max(int(counts[ab, 1:].sum()), int(counts[ba, 1:].sum())))
        if not n:
            continue
        paired += n
        for edge in (ab, ba):
            remaining = n
            for cls in range(7, -1, -1):
                take = min(remaining, int(counts[edge, cls]))
                accept[edge, cls] = take
                remaining -= take
    selected = ordinal < accept.ravel()[keys]
    moved = valid_ids[selected]
    result = o.copy()
    result[moved, best_slot[moved]] = best_target[moved]
    result = result.reshape(owner.shape)
    flow_after, nv_after = edge_counts(result, nodes, gpus)
    assert np.all(flow_after <= flow)
    assert np.all(nv_after <= nv_before)
    assert np.array_equal(np.bincount(result.ravel(), minlength=ranks), np.bincount(owner.ravel(), minlength=ranks))
    assert np.all(support[result, topk])
    assert int(flow.sum() - flow_after.sum()) == int(best_d[moved].sum())
    assert int(nv_before.sum() - nv_after.sum()) == int(best_v[moved].sum())
    return result, {
        "round": step, "candidate_checks": evaluated,
        "candidate_limit_per_pair": 2, "resident_candidates": resident_candidates,
        "positive_candidates": positive_candidates, "proposed_tokens": len(valid_ids),
        "accepted_tokens": len(moved), "exchanges": paired,
        "rdma_copies_removed": int(flow.sum() - flow_after.sum()),
        "nvlink_copies_removed": int(nv_before.sum() - nv_after.sum()),
        "allowed_undirected_edges": len(edges),
        "maximum_rank_degree": int(degrees.max()),
        "histogram_counters": 2 * len(edges) * 8,
        "histogram_int32_bytes": 2 * len(edges) * 8 * 4,
        "dense_allgather_received_count_bytes_per_rank": 2 * len(edges) * 8 * 4 * ranks,
        "token_comparison_sort": False,
        "initial_state_pair_visits": tokens * kmax,
        "post_commit_audit_pair_visits": tokens * kmax,
        "stable_bucket_ordinal_token_visits": len(valid_ids),
        "state_statistics": "O(TK) fused seen/multiple masks and exact deduplicated histograms",
        "post_commit_audit_scope": "reference correctness recomputation; still included in reported CPU time",
        "per_directed_transport_edge_nonincrease": True,
    }


def optimize(topk, owner, support, *, nodes=4, gpus=8, rounds=1, pressure_weighted=True, edges=None):
    start = time.perf_counter()
    result = np.asarray(owner).copy()
    history = []
    for step in range(rounds):
        result, stats = _round(topk, result, support, nodes, gpus, step, pressure_weighted, edges)
        history.append(stats)
    return result, {"rounds": history,
                    "cpu_reference_seconds_not_kernel_latency": time.perf_counter() - start,
                    "scope": "CPU semantic prototype; full GPU planner and collectives unimplemented",
                    "pressure_weighted": pressure_weighted}


def finalize(topk, owner, support, home_rank, budget=8, padding=128):
    """Rebuild unique padded execution rows once, in stable source/pair order."""
    ranks, _, _ = topk.shape
    experts = len(home_rank)
    home_per_rank = experts // ranks
    keys = (owner.astype(np.int64) * experts + topk).ravel()
    counts = np.bincount(keys, minlength=ranks * experts).reshape(ranks, experts)
    padded = ((counts + padding - 1) // padding) * padding
    slots = np.full((ranks, experts), -1, np.int32)
    slot_counts = np.zeros((ranks, home_per_rank + budget), np.int64)
    for rank in range(ranks):
        local = np.flatnonzero(home_rank == rank)
        extras = np.flatnonzero(support[rank] & (home_rank != rank))
        if len(extras) > budget:
            raise ValueError("replica budget exceeded")
        slots[rank, local] = np.arange(len(local))
        slots[rank, extras] = home_per_rank + np.arange(len(extras))
        active = np.flatnonzero(support[rank])
        slot_counts[rank, slots[rank, active]] = counts[rank, active]
    slot_padded = ((slot_counts + padding - 1) // padding) * padding
    cu = np.concatenate((np.zeros((ranks, 1), np.int64), np.cumsum(slot_padded, axis=1)), axis=1)
    starts = cu[np.arange(ranks)[:, None], np.maximum(slots, 0)]
    local_slot = slots[owner, topk]
    assert np.all(local_slot >= 0)
    physical = owner.astype(np.int32) * (home_per_rank + budget) + local_slot
    occurrence = np.zeros(ranks * experts, np.int32)
    row = np.empty(len(keys), np.int32)
    flat_starts = starts.ravel()
    for i, key in enumerate(keys):
        row[i] = flat_starts[key] + occurrence[key]
        occurrence[key] += 1
    return dict(final_pair_counts_rank_expert=counts,
                padded_pair_counts_rank_expert=padded,
                physical_pair_counts=slot_counts,
                physical_cu_seqlens=cu,
                physical_topk=physical,
                destination_row=row.reshape(topk.shape))
