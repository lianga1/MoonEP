"""Independently check two-rank mask proposals/stable block-prefix semantics.

This is a CPU check, not a CUDA timing or GPU validation. It compares every
round against the existing general-rank bucket_optimizer reference.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis/vendor'))
from bucket_optimizer import _run_round


def propose(old0, exec0, exec1, source, token, step):
    best, key, move = -1, 72, 0
    preferred = (token + step) & 3
    ascending = ((token + step) & 1) == 0
    for a in range(2):
        b = 1 - a
        oa = (old0 ^ 255) if a else old0
        ob = oa ^ 255
        available = oa & (exec1 if b else exec0)
        if not available:
            continue
        ca, cb, na = (bin(v).count('1') for v in (oa, ob, available))
        for q in range(4):
            size = (1 << q) if q < 3 else ca
            if q < 3 and size > na:
                continue
            if q == 3 and (na != ca or size in (1, 2, 4)):
                continue
            gain = int(a != source and size == ca) - int(b != source and cb == 0)
            if gain < 0:
                continue
            score = ((gain * 4 + 3 - ((q - preferred + 4) & 3)) * 9 + cb) * 2
            score += int((a < b) == ascending)
            edge = a * 2 + b
            ordered = score * 4 + 3 - edge
            if ordered > best:
                best, key = ordered, (edge * 2 + gain) * 9 + size
                move, bits = 0, available
                for _ in range(size):
                    bit = bits & -bits
                    move |= bit
                    bits ^= bit
    return key, move


def mask_round(routes, old, support, step):
    S = routes.shape[1]
    x = routes.reshape(-1, 8)
    o = old.reshape(-1, 8)
    keys = np.full(2 * S, 72, np.int32)
    moves = np.zeros(2 * S, np.int32)
    ordinal = np.zeros(2 * S, np.int32)
    counts = np.zeros((2, 72), np.int32)
    for rank in range(2):
        blocks = (S + 255) // 256
        bc = np.zeros((72, blocks), np.int32)
        for t in range(S):
            g = rank * S + t
            om = sum((int(o[g, k]) == 0) << k for k in range(8))
            em = [sum(int(support[r, x[g, k]]) << k for k in range(8)) for r in range(2)]
            key, mask = propose(om, *em, rank, g, step)
            keys[g], moves[g] = key, mask
            if key < 72:
                ordinal[g] = bc[key, t // 256]
                bc[key, t // 256] += 1
        bp = np.cumsum(bc, axis=1) - bc
        counts[rank] = bc.sum(axis=1)
        for t in range(S):
            g = rank * S + t
            if keys[g] < 72:
                ordinal[g] += bp[keys[g], t // 256]
    total = counts.sum(axis=0)
    accept = np.zeros(72, np.int32)
    stats = np.zeros(4, np.int64)
    for size in range(1, 9):
        a0, a1, b0, b1 = 18 + size, 27 + size, 36 + size, 45 + size
        x0, x1, y0, y1 = (int(total[k]) for k in (a0, a1, b0, b1))
        n = min(x0 + x1, y0 + y1, max(x1, y1))
        p, q = min(n, x1), min(n, y1)
        accept[a0], accept[a1], accept[b0], accept[b1] = n - p, p, n - q, q
        stats += np.array([n, 2*n*size, 2*n, p+q])
    result = o.copy()
    for g, key in enumerate(keys):
        if key >= 72:
            continue
        prior = counts[0, key] if g >= S else 0
        if ordinal[g] + prior >= accept[key]:
            continue
        dest = (int(key) // 9 // 2) % 2
        for k in range(8):
            if moves[g] & (1 << k):
                result[g, k] = dest
    return result.reshape(old.shape), stats


def main():
    cases = []
    for S in (1, 7, 31, 32, 33, 255, 256, 257, 258, 513):
        for seed in (17, 29):
            rng = np.random.default_rng(seed + S)
            E = 24
            routes = np.stack([rng.choice(E, 8, replace=False) for _ in range(2*S)]).astype(np.int32).reshape(2, S, 8)
            support = rng.random((2, E)) < .65
            support[np.arange(E) % 2, np.arange(E)] = True
            old = rng.integers(0, 2, routes.shape, dtype=np.int32)
            bad = ~support[old, routes]
            old[bad] ^= 1
            resident = support[0].astype(np.int64) + 2 * support[1].astype(np.int64)
            for step in range(4):
                expected = _run_round(routes.reshape(-1, 8), old.reshape(-1, 8), resident, 2, S, 1, False, step)
                result, stats = mask_round(routes, old, support, step)
                assert np.array_equal(result.reshape(-1, 8), expected[0]), (S, seed, step)
                assert stats.tolist() == [int(expected[i]) for i in (6, 8, 9, 10)]
                old = result
            cases.append({'S': S, 'seed': seed, 'rounds': 4, 'exact': True})
    print(json.dumps({'cpu_semantics_only': True, 'cases': cases, 'round_comparisons': len(cases)*4}, indent=2))


if __name__ == '__main__':
    main()
