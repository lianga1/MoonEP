"""Weight-neutral same-expert quota exchange, NumPy semantic reference.

One fixed edge per replicated expert, one proposal per token, and fixed
priority buckets keep pair work O(TK). Equal expert/direction match counts
preserve the complete expert/rank allocation, active weight copies, gradient
routes, padding, and rank compute loads. Payload edges can only disappear.
No GPU latency or distributed prefix/publication implementation is implied.
"""
from __future__ import annotations
import time
import numpy as np

try:
    from .rail_planner import edge_counts
except ImportError:
    from rail_planner import edge_counts


def optimize_weight_neutral(topk, owner, *, nodes=4, gpus=8, rounds=1,
                            pressure_weighted=True):
    """Return owner and diagnostics without introducing or refreshing a copy.

    This preserves *required* copies exactly. Training still refreshes all seed
    replicas and reduces their gradients; those costs are not zero and must be
    charged to baseline and candidate alike by the surrounding cost model.
    """
    started = time.perf_counter()
    x, original = np.asarray(topk), np.asarray(owner)
    if x.shape != original.shape or x.ndim != 3 or x.shape[0] != nodes * gpus:
        raise ValueError("routing shape/topology mismatch")
    if x.dtype.kind not in 'iu' or original.dtype.kind not in 'iu' or x.size == 0:
        raise ValueError("nonempty integer routing required")
    ranks, per_rank, kmax = x.shape
    if ranks > 32 or rounds not in (0, 1, 2):
        raise ValueError("at most 32 ranks and 0/1/2 bounded rounds supported")
    if np.any(original < 0) or np.any(original >= ranks) or np.any(x < 0):
        raise ValueError("invalid rank/expert")
    experts = int(x.max()) + 1
    alloc = np.bincount((original.astype(np.int64) * experts + x).ravel(),
                       minlength=ranks * experts).reshape(ranks, experts)
    result = original.copy()
    xx = x.reshape(-1, kmax)
    tokens = len(xx)
    src = np.repeat(np.arange(ranks), per_rank)
    sn, rail = src // gpus, src % gpus
    # Two highest populated ranks; selecting the small E x R table is O(RE).
    a = np.argmax(alloc, axis=0)
    second = alloc.copy()
    second[a, np.arange(experts)] = 0
    b = np.argmax(second, axis=0)
    active = second[b, np.arange(experts)] > 0
    eids = np.flatnonzero(active)
    edge_of_expert = np.full(experts, -1, np.int32)
    edge_of_expert[eids] = np.arange(len(eids))
    history = []
    for step in range(rounds):
        oo = result.reshape(-1, kmax)
        if not len(eids):
            history.append(dict(round=step, proposed_tokens=0, accepted_tokens=0,
                                reason='home-only or unsplit expert allocation'))
            break
        flow, nv, masks = edge_counts(result, nodes, gpus, return_masks=True)
        seen, single, nseen, nsingle = masks
        ratios = flow.sum(axis=(0, 2)).astype(float)
        ratios /= max(1.0, ratios.mean())
        if not pressure_weighted:
            ratios[:] = 1
        pressure = 1 + 4 * ratios[rail] ** 4
        priority = np.clip(np.rint(2 + 3 * ratios[rail]), 2, 7).astype(np.int16)
        best = np.full(tokens, -np.inf)
        slots = np.full(tokens, -1, np.int16)
        dests = np.zeros(tokens, np.int16)
        classes = np.zeros(tokens, np.int16)
        keys_edge = np.zeros(tokens, np.int32)
        gains_d = np.zeros(tokens, np.int8)
        gains_v = np.zeros(tokens, np.int8)
        eligible_count = 0
        for k in range(kmax):
            e, old = xx[:, k], oo[:, k]
            is_a, is_b = old == a[e], old == b[e]
            dest = np.where(is_a, b[e], a[e])
            rb = np.left_shift(np.uint64(1), old.astype(np.uint64))
            db = np.left_shift(np.uint64(1), dest.astype(np.uint64))
            oldn, newn = old // gpus, dest // gpus
            nb = np.left_shift(np.uint64(1), oldn.astype(np.uint64))
            dnb = np.left_shift(np.uint64(1), newn.astype(np.uint64))
            # Destination must already be reached, or be source GPU itself.
            valid = active[e] & (is_a | is_b) & (((seen & db) != 0) | (dest == src))
            removed = (oldn != sn) & ((nsingle & nb) != 0)
            added = (newn != sn) & ((nseen & dnb) == 0)
            dg = np.where(oldn != newn, removed.astype(np.int8) - added.astype(np.int8), 0)
            vg = (((old % gpus != rail) & ((single & rb) != 0)).astype(np.int8)
                  - ((dest % gpus != rail) & ((seen & db) == 0)).astype(np.int8))
            assert not np.any(valid & ((dg < 0) | (vg < 0)))
            eligible_count += int(valid.sum())
            score = pressure * dg + 0.5 * vg
            # Prefer positive buckets; deterministic slot tie avoids a sort.
            score += ((old * 31 + dest * 13 + k + np.arange(tokens) + step) % 127) / 100000.
            use = valid & (score > best)
            best[use], slots[use], dests[use] = score[use], k, dest[use]
            classes[use] = np.where(dg > 0, priority, np.where(vg > 0, 1, 0))[use]
            keys_edge[use] = (2 * edge_of_expert[e] + is_b.astype(np.int32))[use]
            gains_d[use], gains_v[use] = dg[use], vg[use]
        ids = np.flatnonzero(slots >= 0)
        key = keys_edge[ids] * 8 + classes[ids]
        counts = np.zeros(len(eids) * 16, np.int64)
        ordinal = np.empty(len(ids), np.int32)
        for i, key_i in enumerate(key):
            ordinal[i] = counts[key_i]
            counts[key_i] += 1
        counts = counts.reshape(len(eids), 2, 8)
        accepts = np.zeros_like(counts)
        swaps = 0
        for edge in range(len(eids)):
            n = min(int(counts[edge, 0].sum()), int(counts[edge, 1].sum()),
                    int(counts[edge, :, 1:].sum(axis=1).max()))
            swaps += n
            for direction in range(2):
                remaining = n
                for cls in range(7, -1, -1):
                    take = min(remaining, int(counts[edge, direction, cls]))
                    accepts[edge, direction, cls] = take
                    remaining -= take
        chosen = ids[ordinal < accepts.ravel()[key]]
        changed = oo.copy()
        changed[chosen, slots[chosen]] = dests[chosen]
        result = changed.reshape(original.shape)
        check = np.bincount((result.astype(np.int64) * experts + x).ravel(),
                            minlength=ranks * experts).reshape(ranks, experts)
        assert np.array_equal(check, alloc)
        next_f, next_v = edge_counts(result, nodes, gpus)
        assert np.all(next_f <= flow) and np.all(next_v <= nv)
        assert int(flow.sum() - next_f.sum()) == int(gains_d[chosen].sum())
        assert int(nv.sum() - next_v.sum()) == int(gains_v[chosen].sum())
        history.append(dict(round=step, eligible_pairs=eligible_count,
                            proposed_tokens=len(ids), accepted_tokens=len(chosen),
                            exchanges=swaps, rdma_copies_removed=int(flow.sum()-next_f.sum()),
                            nvlink_copies_removed=int(nv.sum()-next_v.sum()),
                            candidate_checks=tokens*kmax, histogram_counters=len(eids)*16))
    return result, dict(method='weight_neutral_exact_quota', rounds=history,
                        selected_expert_edges=len(eids), new_active_replicas=0,
                        expert_rank_counts_preserved=True,
                        per_directed_transport_edge_nonincrease=True,
                        padding_preserved=True, token_comparison_sort=False,
                        histogram_int32_bytes=len(eids)*16*4,
                        dense_allgather_received_count_bytes_per_rank=len(eids)*16*4*ranks,
                        complexity='O(TK + RE), one candidate per pair, one proposal per token, <=2 rounds',
                        cpu_reference_seconds_not_kernel_latency=time.perf_counter()-started)


def optimize_hierarchical(topk, home_rank, *, budget=8, nodes=4, gpus=8,
                          seed_owner=None, mapping='locality', allocation_order='node_first'):
    """Build balanced quotas with concentrated node-imbalance spill.

    Default node_first settles aggregate node imbalance using large experts,
    then balances ranks within each node. rank_first retains the initial
    ablation that balances local ranks before inter-node residuals. This is
    quota planning order, not a change to the RDMA-then-NVLink token path.
    Weights are still fetched from their home rank. This routine does not omit
    training weight refresh or gradients: it reduces their required routes.
    Two CPU mapping ablations share quotas: source-local-first and stable FIFO.
    """
    started = time.perf_counter()
    x, home = np.asarray(topk), np.asarray(home_rank)
    if x.ndim != 3 or x.shape[0] != nodes*gpus or x.dtype.kind not in 'iu':
        raise ValueError('integer routing shape/topology mismatch')
    ranks, per_rank, kmax = x.shape
    experts = len(home)
    if np.any(home < 0) or np.any(home >= ranks) or np.any(x < 0) or np.any(x >= experts):
        raise ValueError('invalid IDs')
    if budget < 0 or mapping not in ('locality','fifo') or allocation_order not in ('node_first','rank_first'):
        raise ValueError('invalid budget/mapping')
    cap = per_rank*kmax
    source_counts = np.stack([np.bincount(row.ravel(), minlength=experts) for row in x])
    demand = source_counts.sum(0)
    alloc = np.zeros((ranks, experts), np.int64)
    alloc[home, np.arange(experts)] = demand
    balance = alloc.sum(1)-cap
    extra = np.zeros(ranks, np.int64)
    steps = []
    failure = None
    if allocation_order == 'node_first':
        # Settle only the unavoidable aggregate node imbalance first. A donor
        # rank may temporarily become underfull: its node's other ranks will
        # repair that locally. This concentrates cross-node work in fewer full
        # experts instead of copying many tiny per-rank spill remainders.
        node_balance = balance.reshape(nodes,gpus).sum(1)
        while np.any(node_balance > 0):
            choices = []
            for an in np.flatnonzero(node_balance > 0):
                for bn in np.flatnonzero(node_balance < 0):
                    for a in range(int(an)*gpus,(int(an)+1)*gpus):
                        es = np.flatnonzero((home == a) & (alloc[a] > 0))
                        for b in range(int(bn)*gpus,(int(bn)+1)*gpus):
                            if balance[b] >= 0:
                                continue
                            legal = es[(alloc[b,es] > 0) | (extra[b] < budget)]
                            if not len(legal):
                                continue
                            limit = min(int(node_balance[an]),int(-node_balance[bn]),int(-balance[b]))
                            take = np.minimum(alloc[a,legal],limit)
                            reuse_node = alloc[int(bn)*gpus:(int(bn)+1)*gpus,legal].sum(0) > 0
                            reuse_rank = alloc[b,legal] > 0
                            order = np.lexsort((legal,-take,-reuse_rank.astype(np.int8),-reuse_node.astype(np.int8)))
                            q = int(order[0])
                            choices.append((bool(reuse_node[q]),bool(reuse_rank[q]),int(take[q]),-a,-b,-int(legal[q])))
            if not choices:
                break
            _,_,take,na,nb,ne = max(choices)
            a,b,e = -na,-nb,-ne
            if alloc[b,e] == 0:
                extra[b] += 1
            alloc[a,e] -= take
            alloc[b,e] += take
            balance[a] -= take
            balance[b] += take
            node_balance[a//gpus] -= take
            node_balance[b//gpus] += take
            steps.append(dict(phase='inter',source_rank=a,destination_rank=b,expert=e,pairs=take))
    # Each nontrivial iteration either balances one rank or consumes one
    # home expert remainder. The table is tiny; no token-level sort involved.
    for phase in (('intra',) if allocation_order == 'node_first' else ('intra','inter')):
        while np.any(balance > 0):
            choices = []
            for a in np.flatnonzero(balance > 0):
                es = np.flatnonzero((home == a) & (alloc[a] > 0))
                for b in np.flatnonzero(balance < 0):
                    if (a//gpus == b//gpus) != (phase == 'intra'):
                        continue
                    legal = es[(alloc[b,es] > 0) | (extra[b] < budget)]
                    if not len(legal):
                        continue
                    # Reusing an already paid copy is first, then maximize
                    # filled quota to avoid splitting one deficit over copies.
                    take = np.minimum(alloc[a,legal], min(int(balance[a]),int(-balance[b])))
                    existing = alloc[b,legal] > 0
                    order = np.lexsort((legal, -take, -existing.astype(np.int8)))
                    e = int(legal[order[0]])
                    t = int(take[order[0]])
                    choices.append((bool(existing[order[0]]), t, -int(a), -int(b), -e))
            if not choices:
                break
            _, take, neg_a, neg_b, neg_e = max(choices)
            a,b,e = -neg_a,-neg_b,-neg_e
            if alloc[b,e] == 0:
                extra[b] += 1
            alloc[a,e] -= take
            alloc[b,e] += take
            balance[a] -= take
            balance[b] += take
            steps.append(dict(phase=phase,source_rank=a,destination_rank=b,expert=e,pairs=take))
    if np.any(balance):
        failure = 'extra expert budget cannot realize greedy hierarchical quotas'
        if seed_owner is None:
            raise ValueError(failure)
        owner = np.asarray(seed_owner).copy()
        if owner.shape != x.shape:
            raise ValueError('fallback shape mismatch')
        alloc = np.bincount((owner.astype(np.int64)*experts+x).ravel(),minlength=ranks*experts).reshape(ranks,experts)
        support = (np.arange(ranks)[:,None] == home) | (alloc > 0)
        if np.any((support & (np.arange(ranks)[:,None] != home)).sum(1) > budget):
            raise ValueError('fallback seed also exceeds budget')
        if not np.all(alloc.sum(1) == cap):
            raise ValueError('fallback seed is unbalanced')
        return owner,support,dict(method='hierarchical_local_first',fallback_reason=failure,
                                 cpu_reference_seconds_not_kernel_latency=time.perf_counter()-started)
    assert np.all(alloc.sum(1) == cap) and np.array_equal(alloc.sum(0),demand)
    support = (np.arange(ranks)[:,None] == home) | (alloc > 0)
    assert np.all((support & (np.arange(ranks)[:,None] != home)).sum(1) <= budget)
    # Source/expert -> destination quotas. At most O(RE) nonzero intervals;
    # this dense CPU array is not the proposed kernel's storage representation.
    routes = np.zeros((ranks,experts,ranks),np.int64)
    if mapping == 'locality':
        for e in range(experts):
            left = source_counts[:,e].copy()
            right = alloc[:,e].copy()
            self_take = np.minimum(left,right)
            routes[np.arange(ranks),e,np.arange(ranks)] = self_take
            left -= self_take
            right -= self_take
            for phase in ('intra','inter'):
                for src in range(ranks):
                    eligible = np.flatnonzero((np.arange(ranks)//gpus == src//gpus)
                                              if phase == 'intra' else (np.arange(ranks)//gpus != src//gpus))
                    for dst in eligible:
                        take = min(int(left[src]),int(right[dst]))
                        routes[src,e,dst] += take
                        left[src] -= take
                        right[dst] -= take
                        if not left[src]:
                            break
            assert not np.any(left) and not np.any(right)
    else:
        for e in range(experts):
            right = alloc[:,e].copy()
            dst = 0
            for src in range(ranks):
                left = int(source_counts[src,e])
                while left:
                    while not right[dst]:
                        dst += 1
                    take = min(left,int(right[dst]))
                    routes[src,e,dst] = take
                    left -= take
                    right[dst] -= take
            assert not np.any(right)
    prefix = routes.cumsum(2)
    owner = np.empty(x.shape,np.int16)
    # Stable bucket ordinals, one O(TK) walk, then small-prefix lookup. A
    # parallel implementation uses block histograms/prefixes, not this CPU loop.
    for src in range(ranks):
        counter = np.zeros(experts,np.int64)
        flat = x[src].ravel()
        ordinal = np.empty(len(flat),np.int64)
        for i,e in enumerate(flat):
            ordinal[i] = counter[e]
            counter[e] += 1
        # Fixed binary-search loop: ceil(log2(R))+1 probes/pair (6 at R=32).
        lo = np.zeros(len(flat),np.int16)
        hi = np.full(len(flat),ranks,np.int16)
        for _ in range((ranks-1).bit_length()+1):
            valid = lo < hi
            mid = (lo+hi)//2
            after = ordinal >= prefix[src,flat,np.minimum(mid,ranks-1)]
            lo = np.where(valid & after,mid+1,lo)
            hi = np.where(valid & ~after,mid,hi)
        owner[src] = lo.reshape(per_rank,kmax)
    got = np.bincount((owner.astype(np.int64)*experts+x).ravel(),minlength=ranks*experts).reshape(ranks,experts)
    assert np.array_equal(got,alloc)
    active_extra = (alloc > 0) & (np.arange(ranks)[:,None] != home)
    cross = active_extra & (np.arange(ranks)[:,None]//gpus != home//gpus)
    return owner,support,dict(method='hierarchical_local_first',mapping=mapping,allocation_order=allocation_order,
                             fallback_reason=None,allocation_steps=steps,quota_transfer_steps=len(steps),
                             source_prefix_probe_passes=(ranks-1).bit_length()+1,
                             intra_pairs_moved=sum(s['pairs'] for s in steps if s['phase']=='intra'),
                             inter_pairs_moved=sum(s['pairs'] for s in steps if s['phase']=='inter'),
                             active_extra_replicas=int(active_extra.sum()),
                             active_cross_node_replicas=int(cross.sum()),
                             max_extra_replicas=int(active_extra.sum(1).max(initial=0)),
                             candidate_full_pair_passes=1,
                             token_comparison_sort=False,
                             nonzero_source_expert_destination_intervals=int(np.count_nonzero(routes)),
                             complexity='O(TK) demand scan; <=2R+E+2N quota steps, each <=O(R^2*(E/R)) home-expert small-table selection; O(E*R^2) locality table; O(TK log R) stable ordinal mapping',
                             cpu_reference_seconds_not_kernel_latency=time.perf_counter()-started)
