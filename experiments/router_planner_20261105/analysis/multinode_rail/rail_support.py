"""Bounded, rail-aware reciprocal expert residency selection (CPU reference).

This module only chooses eligibility. It does not alter execution assignments,
claim that sampled/counting gains will all be accepted, or estimate GPU time.
All seed-required copies are retained; newly selected copies share the same B
budget. Two candidate destinations per pair keep token work linear in T*K.
"""
from __future__ import annotations

import time
import numpy as np


def _masks(owner):
    tokens, k = owner.shape
    seen = np.zeros(tokens, dtype=np.uint64)
    multiple = np.zeros(tokens, dtype=np.uint64)
    for slot in range(k):
        bit = np.left_shift(np.uint64(1), owner[:, slot].astype(np.uint64))
        multiple |= seen & bit
        seen |= bit
    return seen, seen & ~multiple


def select_support(topk, owner, home_rank, budget=8, nodes=4, gpus=8,
                   degree=4, experts_per_edge=2):
    """Return (support[R,E], diagnostics), without modifying input arrays.

    Votes consider only source GPU and one source-node already-visited GPU.
    Rank/node singleton masks give exact single-pair payload-copy deltas.
    The vote table is target rank x active (expert,seed-owner) slots, at most
    R*(E+R*B) entries for a legal seed. A one-pass greedy small-table selection
    chooses positive reciprocal edges with degree and per-direction copy caps.
    New copies of one expert within the same node are deliberately suppressed.

    Complexity: O(T*K) voting and mask construction; O(R*A) vote storage for
    A<=E+R*B active slots; O(R^2 * (E/R+B) log(E/R+B)) small-table selection.
    All sorting is over rank/expert tables, never over tokens. NumPy temporary
    arrays in this semantic reference are not a GPU memory-layout proposal.
    """
    started = time.perf_counter()
    x, o, home = np.asarray(topk), np.asarray(owner), np.asarray(home_rank)
    if x.ndim != 3 or o.shape != x.shape or x.dtype.kind not in "iu" or o.dtype.kind not in "iu":
        raise ValueError("integer topk and owner must have shape [R,S,K]")
    ranks, per_source, k = x.shape
    experts = len(home)
    if ranks != nodes * gpus or ranks > 63 or not 1 <= gpus <= 8:
        raise ValueError("requires R=nodes*gpus<=63 and 1<=gpus<=8")
    if budget < 0 or degree < 0 or experts_per_edge < 0:
        raise ValueError("budgets must be nonnegative")
    if np.any(x < 0) or np.any(x >= experts) or np.any(o < 0) or np.any(o >= ranks):
        raise ValueError("out of range expert or owner")
    if np.any(home < 0) or np.any(home >= ranks):
        raise ValueError("out of range expert home")
    flat_x = x.reshape(-1, k)
    flat_o = o.reshape(-1, k)
    tokens = len(flat_x)
    source = np.repeat(np.arange(ranks), per_source)
    source_node, source_rail = source // gpus, source % gpus
    counts = np.bincount((o.astype(np.int64) * experts + x).ravel(),
                         minlength=ranks * experts).reshape(ranks, experts)
    home_support = np.arange(ranks)[:, None] == home[None]
    support = home_support | (counts > 0)
    extras = np.sum(support & ~home_support, axis=1).astype(np.int64)
    if extras.max(initial=0) > budget:
        raise ValueError("seed execution already exceeds the hard extra-expert budget")
    mandatory = extras.copy()
    active_owner, active_expert = np.nonzero(counts)
    active_slots = len(active_owner)
    slot_lut = np.full((ranks, experts), -1, dtype=np.int32)
    slot_lut[active_owner, active_expert] = np.arange(active_slots)
    pair_slot = slot_lut[flat_o, flat_x]
    owner_slots = [np.flatnonzero(active_owner == a) for a in range(ranks)]

    rank_seen, rank_single = _masks(flat_o)
    node_o = flat_o // gpus
    node_seen, node_single = _masks(node_o)
    flow = np.zeros((nodes, gpus, nodes), dtype=np.int64)
    for target_node in range(nodes):
        present = ((node_seen & np.uint64(1 << target_node)) != 0) & (source_node != target_node)
        flow[:, :, target_node] = np.bincount(source[present], minlength=ranks).reshape(nodes, gpus)
    rail_bytes = flow.sum(axis=(0, 2)).astype(np.float64)
    tx, rx = flow.sum(axis=2), flow.sum(axis=0).T
    rail_ratio = rail_bytes / max(1.0, rail_bytes.mean())
    tx_ratio = tx / max(1.0, tx.mean())
    rx_ratio = rx / max(1.0, rx.mean())
    # RDMA's nominal 4x value plus quadratic pressure emphasizes hot rails;
    # these are selection weights, not hardware latency measurements.
    rdma_value = 4.0 * (1.0 + rail_ratio[source_rail, None] ** 2
                       + tx_ratio[source_node, source_rail, None] ** 2
                       + rx_ratio[node_o, source_rail[:, None]] ** 2)
    old_rank_bit = np.left_shift(np.uint64(1), flat_o.astype(np.uint64))
    old_node_bit = np.left_shift(np.uint64(1), node_o.astype(np.uint64))
    rdma_gain = ((node_o != source_node[:, None])
                 & ((node_single[:, None] & old_node_bit) != 0))
    nv_gain = ((flat_o % gpus != source_rail[:, None])
               & ((rank_single[:, None] & old_rank_bit) != 0))
    values = rdma_gain * rdma_value + nv_gain.astype(np.float64)

    vote = np.zeros((ranks, active_slots), dtype=np.float64)
    opportunities = np.zeros((ranks, active_slots), dtype=np.int64)
    rdma_opportunities = np.zeros((ranks, active_slots), dtype=np.int64)
    targets_considered = 0
    positive_votes = 0

    def add_votes(target):
        nonlocal targets_considered, positive_votes
        legal = (target >= 0) & (target != flat_o)
        targets_considered += int(legal.sum())
        good = legal & (values > 0)
        positive_votes += int(good.sum())
        ids = target[good] * active_slots + pair_slot[good]
        vote[:] += np.bincount(ids, weights=values[good], minlength=ranks * active_slots).reshape(vote.shape)
        opportunities[:] += np.bincount(ids, minlength=ranks * active_slots).reshape(vote.shape)
        rdma_opportunities[:] += np.bincount(ids, weights=rdma_gain[good].astype(np.int64),
                                             minlength=ranks * active_slots).reshape(vote.shape).astype(np.int64)

    # Candidate 1: source GPU. Its local rank equals the source rail, hence
    # no new NVLink copy even when that GPU was not previously visited.
    add_votes(np.broadcast_to(source[:, None], flat_o.shape))
    # Candidate 2: one already-visited source-node rank distinct from source
    # and current owner. Rotate an 8-bit mask, then one ctz table lookup. No
    # scan over all R ranks or top-k subsets is needed per pair.
    node_mask = np.uint64((1 << gpus) - 1)
    local_seen = (rank_seen >> (source_node * gpus).astype(np.uint64)) & node_mask
    local_seen &= ~np.left_shift(np.uint64(1), source_rail.astype(np.uint64))
    pair_local = np.broadcast_to(local_seen[:, None], flat_o.shape).copy()
    same_node = node_o == source_node[:, None]
    pair_local &= np.where(same_node, ~np.left_shift(np.uint64(1), (flat_o % gpus).astype(np.uint64)),
                           np.uint64(0xffffffffffffffff))
    hashed = (np.arange(tokens, dtype=np.uint64)[:, None] * np.uint64(0x9e3779b1)
              + flat_x.astype(np.uint64) * np.uint64(0x85ebca6b)) % np.uint64(gpus)
    rotated = ((pair_local >> hashed) | (pair_local << (np.uint64(gpus) - hashed))) & node_mask
    ctz = np.zeros(1 << gpus, dtype=np.int16)
    for value in range(1, len(ctz)):
        ctz[value] = (value & -value).bit_length() - 1
    picked = (ctz[rotated.astype(np.int64)] + hashed.astype(np.int64)) % gpus
    second = source_node[:, None] * gpus + picked
    second[rotated == 0] = -1
    add_votes(second)
    voting_done = time.perf_counter()

    node_support = support.reshape(nodes, gpus, experts).any(axis=1)
    rank_degrees = np.zeros(ranks, dtype=np.int32)

    def directional(a, b):
        """At most experts_per_edge eligible expert slots; tiny table only."""
        slots = owner_slots[a]
        order = slots[np.lexsort((active_expert[slots], -vote[b, slots]))]
        chosen = []
        new = 0
        for q in order:
            if vote[b, q] <= 0:
                break
            e = int(active_expert[q])
            missing = not support[b, e]
            if missing:
                if new >= experts_per_edge or extras[b] + new >= budget:
                    continue
                if node_support[b // gpus, e]:
                    continue
                new += 1
            chosen.append(int(q))
            if len(chosen) == experts_per_edge:
                break
        return chosen, new

    def package(a, b):
        ab, add_b = directional(a, b)
        ba, add_a = directional(b, a)
        if not ab or not ba or add_a + add_b == 0:
            return None
        n_ab, n_ba = int(opportunities[b, ab].sum()), int(opportunities[a, ba].sum())
        if not n_ab or not n_ba:
            return None
        matched = min(n_ab, n_ba)
        utility = matched * (vote[b, ab].sum() / n_ab + vote[a, ba].sum() / n_ba)
        return float(utility / (add_a + add_b)), ab, ba, matched

    packages = []
    if degree and experts_per_edge:
        for a in range(ranks):
            for b in range(a + 1, ranks):
                if a // gpus == b // gpus:
                    continue
                proposal = package(a, b)
                if proposal is not None:
                    packages.append((proposal[0], a, b))
    packages.sort(key=lambda item: (-item[0], item[1], item[2]))
    accepted = []
    for _, a, b in packages:
        if rank_degrees[a] >= degree or rank_degrees[b] >= degree:
            continue
        proposal = package(a, b)
        if proposal is None:
            continue
        value, ab, ba, matched = proposal
        added_a, added_b = [], []
        for target, slots, added in ((b, ab, added_b), (a, ba, added_a)):
            for q in slots:
                e = int(active_expert[q])
                if not support[target, e]:
                    support[target, e] = True
                    node_support[target // gpus, e] = True
                    extras[target] += 1
                    added.append(e)
        rank_degrees[a] += 1
        rank_degrees[b] += 1
        accepted.append({"a": a, "b": b, "add_at_a": added_a, "add_at_b": added_b,
                         "estimated_matched_pair_opportunities": matched,
                         "selection_utility_per_new_copy": value,
                         "ab_rdma_reducing_votes": int(rdma_opportunities[b, ab].sum()),
                         "ba_rdma_reducing_votes": int(rdma_opportunities[a, ba].sum())})

    assert np.all(support | ~home_support)
    assert np.all(support[flat_o, flat_x])
    assert np.all(extras <= budget) and np.all(rank_degrees <= degree)
    support_mask = np.bitwise_or.reduce(np.where(support.T,
        np.left_shift(np.uint64(1), np.arange(ranks, dtype=np.uint64))[None], np.uint64(0)), axis=1)
    alternatives = support_mask[flat_x] & ~old_rank_bit
    source_node_bits = node_mask << (source_node * gpus).astype(np.uint64)
    source_gpu_bits = np.left_shift(np.uint64(1), source.astype(np.uint64))
    source_gpu_candidate = (alternatives & source_gpu_bits[:, None]) != 0
    local_visited_candidate = (alternatives & source_node_bits[:, None] & rank_seen[:, None]) != 0
    local_candidate = source_gpu_candidate | local_visited_candidate
    visited_candidate = (alternatives & rank_seen[:, None]) != 0
    eligible = local_candidate | visited_candidate
    end = time.perf_counter()
    diagnostics = {
        "method": "bounded_reciprocal_source_node_votes",
        "scope": "CPU residency reference; candidate votes are estimates, not committed execution changes",
        "budget": budget, "degree_cap": degree, "experts_per_direction_cap": experts_per_edge,
        "mandatory_extra_per_rank": mandatory.tolist(), "selected_extra_per_rank": extras.tolist(),
        "new_extra_per_rank": (extras - mandatory).tolist(), "selected_rank_degrees": rank_degrees.tolist(),
        "active_seed_slots": active_slots, "vote_table_entries": ranks * active_slots,
        "positive_candidate_votes": positive_votes, "candidate_target_checks": targets_considered,
        "max_targets_per_pair": 2, "candidate_rank_edges": len(packages), "accepted_edges": accepted,
        "pairs_with_source_node_alternative": int(local_candidate.sum()),
        "pairs_with_source_gpu_alternative": int(source_gpu_candidate.sum()),
        "pairs_with_visited_rank_alternative": int(visited_candidate.sum()),
        "pairs_with_any_bounded_target": int(eligible.sum()),
        "tokens_with_any_bounded_target": int(eligible.any(axis=1).sum()),
        "initial_rail_copies": rail_bytes.astype(np.int64).tolist(),
        "selection_weights": "RDMA 4*(1+rail_ratio^2+TX_ratio^2+RX_ratio^2), NVLink 1; heuristic, not measured time",
        "new_same_node_duplicate_experts_suppressed": True,
        "candidate_estimate_limits": "votes can overlap across token pairs and edges; one-proposal matching must validate realized gains",
        "cpu_wall_seconds_not_gpu_time": {"mask_and_voting": voting_done - started,
                                           "small_table_and_diagnostics": end - voting_done, "total": end - started},
    }
    return support, diagnostics
