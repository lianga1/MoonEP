"""CPU/Numba semantic model of fixed-residency, bucketed pair exchanges.

This file implements no GPU kernel and its wall times are NOT GPU estimates.
Each token proposes at most once in a round.  Opposite directions match equal
bundle sizes, so all accepted proposals can commit simultaneously without
changing any rank's load.  Residency is never modified; the caller guarantees
that support includes home and at most eight additional experts per rank.

``bundle`` considers canonical subsets of sizes 1, 2, 4, plus the entire old
destination when all its pairs are executable at the target.  It does not
enumerate all subsets or solve a planning problem.  A token/round-dependent
size preference prevents always preferring the largest bundle.  ``bundle_max``
retains the older maximal-eligible-bundle heuristic as an extra control mode.

Complexity for T tokens, K choices, R ranks, M=T*K:
  J = sum_e frequency[e] * (resident_rank_count[e] - 1).
Single mode: O(M + J + T*R), plus O(T + R^2) matching/commit.
Bundle modes: O(M + J + T*(R^2 + 4*min(J/T,R^2)*K)) as a conservative bound,
because this simple reference reconstructs up to four <=K-bit subset masks
per nonempty edge.  For fixed K<=32,R<=8 these are small constants; no claim
that arbitrary bundle search is asymptotically cheaper is made.  The candidate
matrix is R^2 per token and is reused, not stored for all tokens.  Proposals
are O(T), histogram storage O(R^2*K), input/output O(M).

Independent small-input audit:
  .venv/bin/python experiments/moe_placement/hotcore128k/bucket_optimizer.py
"""

from __future__ import annotations

import json
import time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


@njit(cache=True)
def _ctz(value):
    out = 0
    while (value & 1) == 0:
        value >>= 1
        out += 1
    return out


@njit(cache=True)
def _popcount(value):
    out = 0
    while value:
        value &= value - 1
        out += 1
    return out


@njit(cache=True)
def _first_bits(mask, size):
    """One canonical subset, not an exponential search over subsets."""
    selected = 0
    for _ in range(size):
        bit = mask & -mask
        selected |= bit
        mask ^= bit
    return selected


@njit(cache=True)
def _edge_tie(a, b, token, step, ranks):
    lo, hi = min(a, b), max(a, b)
    pair_count = ranks * (ranks - 1) // 2
    pair = lo * (2 * ranks - lo - 1) // 2 + hi - lo - 1
    priority = pair_count - 1 - ((pair - token - 17 * step) % pair_count)
    prefer_ascending = ((token // pair_count + step) & 1) == 0
    return 2 * priority + int((a < b) == prefer_ascending)


@njit(cache=True)
def _run_round(experts, old, resident, ranks, per_source, variant,
               allow_neutral, step, token_base=0):
    """variant: 0=single, 1=multiple bundle sizes, 2=old maximal bundle."""
    tokens, topk = experts.shape
    max_size = 1 if variant == 0 else topk
    stride = max_size + 1
    counts = np.zeros(ranks * ranks * 2 * stride, np.int64)
    accept = np.zeros_like(counts)
    keys = np.full(tokens, -1, np.int32)
    masks = np.zeros(tokens, np.int64)
    ordinals = np.zeros(tokens, np.int64)
    occupancy = np.zeros(ranks, np.int64)
    eligible = np.zeros((ranks, ranks), np.int64)
    sizes = np.zeros(4, np.int64)
    edge_span = max(1, ranks * (ranks - 1))
    target_checks = 0
    candidate_checks = 0
    legal_candidates = 0
    proposed = 0
    proposed_gain_one = 0
    candidate_sizes = np.zeros(topk + 1, np.int64)
    proposal_sizes = np.zeros(topk + 1, np.int64)

    for t in range(tokens):
        occupancy[:] = 0
        for k in range(topk):
            occupancy[old[t, k]] += 1
        source = (t + token_base) // per_source
        score_best, a_best, b_best, gain_best = -1, -1, -1, -1
        size_best, mask_best = 0, 0

        if variant == 0:
            for k in range(topk):
                a = old[t, k]
                targets = resident[experts[t, k]] & ~(1 << a)
                while targets:
                    b = _ctz(targets)
                    targets &= targets - 1
                    target_checks += 1
                    candidate_checks += 1
                    gain = int(a != source and occupancy[a] == 1)
                    gain -= int(b != source and occupancy[b] == 0)
                    if gain < 0:
                        continue
                    legal_candidates += 1
                    candidate_sizes[1] += 1
                    # In the explicitly neutral round, prefer source-local
                    # destinations on an equal immediate gain.  Otherwise a
                    # symmetric 0+0 exchange can move local pairs outward and
                    # create larger remote bundles without reducing traffic.
                    # Positive-only single mode retains the former ordering.
                    local_tie = int(allow_neutral and b == source)
                    gain_span = 2 if allow_neutral else 1
                    score = (((gain * gain_span + local_tie) * (topk + 1)
                              + occupancy[b]) * edge_span
                             + _edge_tie(a, b, t + token_base, step, ranks))
                    if score > score_best:
                        score_best, a_best, b_best = score, a, b
                        gain_best, size_best, mask_best = gain, 1, 1 << k
        else:
            eligible[:, :] = 0
            for k in range(topk):
                a = old[t, k]
                targets = resident[experts[t, k]] & ~(1 << a)
                while targets:
                    b = _ctz(targets)
                    targets &= targets - 1
                    target_checks += 1
                    eligible[a, b] |= 1 << k
            for a in range(ranks):
                for b in range(ranks):
                    available = eligible[a, b]
                    if available == 0:
                        continue
                    available_size = _popcount(available)
                    count_sizes = 0
                    if variant == 2:
                        sizes[0] = available_size
                        count_sizes = 1
                    else:
                        # Explicitly consider small blocks even when a larger
                        # eligible block exists; opposite directions may have
                        # different maximal sizes.  Deduplicate full size.
                        for q in range(3):
                            size = 1 << q
                            if size <= available_size:
                                sizes[count_sizes] = size
                                count_sizes += 1
                        if available_size == occupancy[a]:
                            full = occupancy[a]
                            if full != 1 and full != 2 and full != 4:
                                sizes[count_sizes] = full
                                count_sizes += 1
                    for q in range(count_sizes):
                        size = sizes[q]
                        mask = _first_bits(available, size)
                        candidate_checks += 1
                        gain = int(a != source and size == occupancy[a])
                        gain -= int(b != source and occupancy[b] == 0)
                        if gain < 0:
                            continue
                        legal_candidates += 1
                        candidate_sizes[size] += 1
                        if variant == 2:
                            # Preserve the maximal-bundle control's old order.
                            size_tie = size
                            size_span = topk + 1
                        else:
                            size_class = 0 if size == 1 else (1 if size == 2 else
                                         (2 if size == 4 else 3))
                            # Rotating among the four size classes prevents
                            # enumeration degenerating into always-largest.
                            preferred = ((t + token_base) // max(1, ranks * (ranks - 1) // 2)
                                         + step) % 4
                            size_tie = 3 - ((size_class - preferred) % 4)
                            size_span = 4
                        score = (((gain * size_span + size_tie) * (topk + 1)
                                  + occupancy[b]) * edge_span
                                 + _edge_tie(a, b, t + token_base, step, ranks))
                        if score > score_best:
                            score_best, a_best, b_best = score, a, b
                            gain_best, size_best, mask_best = gain, size, mask

        if a_best >= 0:
            key = (((a_best * ranks + b_best) * 2 + gain_best) * stride
                   + size_best)
            keys[t], masks[t], ordinals[t] = key, mask_best, counts[key]
            counts[key] += 1
            proposed += 1
            proposed_gain_one += gain_best
            proposal_sizes[size_best] += 1

    exchanges = 0
    neutral_exchanges = 0
    predicted_gain = 0
    accepted_sizes = np.zeros(topk + 1, np.int64)
    for a in range(ranks):
        for b in range(a + 1, ranks):
            for size in range(1, max_size + 1):
                ab0 = ((a * ranks + b) * 2) * stride + size
                ab1 = ab0 + stride
                ba0 = ((b * ranks + a) * 2) * stride + size
                ba1 = ba0 + stride
                n = min(counts[ab0] + counts[ab1], counts[ba0] + counts[ba1])
                if not allow_neutral:
                    n = min(n, max(counts[ab1], counts[ba1]))
                accept[ab1] = min(n, counts[ab1])
                accept[ab0] = n - accept[ab1]
                accept[ba1] = min(n, counts[ba1])
                accept[ba0] = n - accept[ba1]
                exchanges += n
                neutral_exchanges += n - max(accept[ab1], accept[ba1])
                predicted_gain += accept[ab1] + accept[ba1]
                accepted_sizes[size] += n

    result = old.copy()
    moved_pairs = 0
    accepted_tokens = 0
    for t in range(tokens):
        key = keys[t]
        if key < 0 or ordinals[t] >= accept[key]:
            continue
        b = (key // stride // 2) % ranks
        mask = masks[t]
        accepted_tokens += 1
        while mask:
            k = _ctz(mask)
            mask &= mask - 1
            result[t, k] = b
            moved_pairs += 1
    return (result, proposed, proposed_gain_one, target_checks, candidate_checks,
            legal_candidates, exchanges, neutral_exchanges, moved_pairs,
            accepted_tokens, predicted_gain, candidate_sizes, proposal_sizes,
            accepted_sizes)


def _metrics(dst, per_source, ranks):
    tokens, topk = dst.shape
    masks = np.zeros(tokens, np.uint16)
    for k in range(topk):
        masks |= np.left_shift(np.uint16(1), dst[:, k].astype(np.uint16))
    source = np.arange(tokens) // per_source
    remote = masks & ~np.left_shift(np.uint16(1), source.astype(np.uint16))
    pop = np.array([bin(x).count('1') for x in range(1 << ranks)])
    return {'remote_destinations': int(pop[remote].sum()),
            'all_destinations': int(pop[masks].sum()),
            'loads': np.bincount(dst.ravel(), minlength=ranks).tolist()}


def optimize(routes, seed_dst, support, rounds=4, mode='single'):
    """Return (dst, stats), preserving every seed rank's exact load.

    routes/seed_dst have shape [R,S,K]; first dimension is source rank.
    support is fixed bool [R,E].  Main modes are single, bundle,
    neutral_then_single, hybrid.  Additional bundle_max is the former
    maximal-eligible-bundle control.  hybrid uses single+neutral in round 1,
    then bundle, single, bundle, ...; neutral_then_single permits neutral
    exchanges only in its first round.  All other rounds require positive
    combined communication gain.  Every execution has a fixed finite limit.
    """
    valid_modes = ('single', 'bundle', 'neutral_then_single', 'hybrid', 'bundle_max')
    if mode not in valid_modes:
        raise ValueError('mode must be one of ' + ', '.join(valid_modes))
    routes, seed_dst = np.asarray(routes), np.asarray(seed_dst)
    support = np.asarray(support, dtype=np.bool_)
    if routes.ndim != 3 or routes.shape != seed_dst.shape:
        raise ValueError('routes and seed_dst must have identical [R,S,K] shapes')
    ranks, per_source, topk = routes.shape
    if not (1 <= ranks <= 8 and per_source > 0 and 1 <= topk <= 32):
        raise ValueError('reference supports 1<=R<=8, S>0, 1<=K<=32')
    if support.ndim != 2 or support.shape[0] != ranks:
        raise ValueError('support must have shape [R,E]')
    if not np.issubdtype(routes.dtype, np.integer) or not np.issubdtype(seed_dst.dtype, np.integer):
        raise TypeError('expert IDs and destination ranks must be integers')
    if int(rounds) != rounds or rounds < 0:
        raise ValueError('rounds must be a nonnegative integer')
    experts = np.ascontiguousarray(routes.reshape(-1, topk), dtype=np.int32)
    dst = np.ascontiguousarray(seed_dst.reshape(-1, topk), dtype=np.int32).copy()
    if np.any(experts < 0) or np.any(experts >= support.shape[1]):
        raise ValueError('expert outside support')
    if np.any(dst < 0) or np.any(dst >= ranks):
        raise ValueError('destination outside [0,R)')
    if not np.all(support[dst, experts]):
        raise ValueError('seed contains unsupported execution')
    if topk > 1 and np.any(np.diff(np.sort(experts, axis=1), axis=1) == 0):
        raise ValueError('top-k experts must be distinct per token')
    resident = np.zeros(support.shape[1], np.int64)
    for r in range(ranks):
        resident |= support[r].astype(np.int64) << r
    initial = _metrics(dst, per_source, ranks)
    previous = initial
    records = []
    begin = time.perf_counter()
    for step in range(int(rounds)):
        allow_neutral = step == 0 and mode in ('neutral_then_single', 'hybrid')
        variant = 0
        if mode == 'bundle' or (mode == 'hybrid' and step % 2 == 1):
            variant = 1
        elif mode == 'bundle_max':
            variant = 2
        out = _run_round(experts, dst, resident, ranks, per_source, variant,
                         allow_neutral, step)
        (dst, proposed, gain_one, checks, candidates, legal, exchanges, neutral,
         moved, accepted_tokens, predicted, candidate_sizes, proposal_sizes,
         accepted_sizes) = out
        current = _metrics(dst, per_source, ranks)
        actual = previous['remote_destinations'] - current['remote_destinations']
        if current['loads'] != initial['loads']:
            raise AssertionError('load vector changed')
        if actual != predicted or actual < 0:
            raise AssertionError('predicted gain differs from true communication')
        if not allow_neutral and neutral:
            raise AssertionError('unexpected neutral exchange')
        records.append({
            'round': step + 1,
            'proposal_mode': ('single', 'bundle', 'bundle_max')[variant],
            'allow_neutral': allow_neutral, 'proposed_tokens': int(proposed),
            'proposed_gain_one': int(gain_one), 'target_checks': int(checks),
            'bundle_matrix_entries': int(experts.shape[0] * ranks * ranks if variant else 0),
            'candidate_gain_checks': int(candidates), 'legal_candidates': int(legal),
            'candidate_sizes': candidate_sizes.tolist(),
            'proposal_sizes': proposal_sizes.tolist(),
            'accepted_exchanges_by_size': accepted_sizes.tolist(),
            'accepted_exchanges': int(exchanges), 'neutral_exchanges': int(neutral),
            'moved_pairs': int(moved), 'accepted_tokens': int(accepted_tokens),
            'communication_reduction': int(actual),
            'remote_destinations': current['remote_destinations'],
        })
        previous = current
    seconds = time.perf_counter() - begin
    if not np.all(support[dst, experts]):
        raise AssertionError('unsupported final execution')
    denominator = initial['remote_destinations']
    stats = {
        'mode': mode, 'engine': 'numba_cpu' if HAVE_NUMBA else 'python_cpu',
        'tokens': int(experts.shape[0]), 'ranks': ranks, 'topk': topk,
        'requested_rounds': int(rounds), 'initial': initial, 'final': previous,
        'communication_reduction': denominator - previous['remote_destinations'],
        'reduction_fraction': ((denominator - previous['remote_destinations']) / denominator
                               if denominator else 0.0),
        'load_vector_preserved': True, 'all_pairs_executable': True,
        'residency_changed': False, 'rounds': records,
        'cpu_seconds_including_possible_jit_and_audits': seconds,
        'timing_scope': 'CPU semantic execution; NOT GPU latency',
    }
    return dst.reshape(routes.shape), stats


def _self_test():
    def independent_remote(dst):
        return sum(len(set(map(int, token)) - {source})
                   for source, batch in enumerate(dst) for token in batch)

    # Same-size complete bundles remove remote destinations that singleton
    # moves alone cannot remove.  No mutation of initial arrays is permitted.
    routes = np.tile(np.arange(4, dtype=np.int32), (2, 1, 1))
    seed = np.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=np.int32)
    support = np.ones((2, 4), bool)
    fixtures = {}
    for mode in ('single', 'bundle', 'neutral_then_single', 'hybrid', 'bundle_max'):
        dst, stats = optimize(routes, seed, support, rounds=3, mode=mode)
        assert independent_remote(dst) == stats['final']['remote_destinations']
        fixtures[mode] = stats['communication_reduction']
    assert fixtures['single'] == 0
    assert fixtures['bundle'] == 2
    assert fixtures['neutral_then_single'] == 2
    assert fixtures['hybrid'] == 2

    rng = np.random.default_rng(20261015)
    tested = 0
    for ranks, per_source, topk in ((2, 13, 4), (4, 17, 8), (8, 19, 16)):
        experts = 32
        routes = np.empty((ranks, per_source, topk), np.int32)
        for token in routes.reshape(-1, topk):
            token[:] = rng.choice(experts, topk, replace=False)
        support = rng.random((ranks, experts)) < .55
        support[np.arange(experts) % ranks, np.arange(experts)] = True
        seed = np.empty_like(routes)
        for token, owners in zip(routes.reshape(-1, topk), seed.reshape(-1, topk)):
            for k, e in enumerate(token):
                owners[k] = rng.choice(np.flatnonzero(support[:, e]))
        originals = routes.copy(), seed.copy(), support.copy()
        for mode in ('single', 'bundle', 'neutral_then_single', 'hybrid', 'bundle_max'):
            dst, stats = optimize(routes, seed, support, rounds=4, mode=mode)
            assert independent_remote(dst) == stats['final']['remote_destinations']
            assert independent_remote(dst) <= independent_remote(seed)
            assert np.array_equal(np.bincount(dst.ravel(), minlength=ranks),
                                  np.bincount(seed.ravel(), minlength=ranks))
            assert np.all(support[dst, routes])
            assert all(np.array_equal(a, b) for a, b in zip((routes, seed, support), originals))
            tested += 1
    return {'ok': True, 'random_mode_cases': tested, 'fixture_gains': fixtures,
            'engine': 'numba_cpu' if HAVE_NUMBA else 'python_cpu'}


if __name__ == '__main__':
    print(json.dumps(_self_test(), indent=2))
