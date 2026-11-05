"""Independent CPU check of CUDA proposal scoring and count-prefix commit protocol.

This validates semantics only; it neither compiles nor measures the CUDA code.
"""
from pathlib import Path
import sys, json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
from vendor.bucket_sharded import _run_round

BUCKETS = 1152


def propose(x, owner, support, per_source, token_base, step):
    n, k = x.shape
    keys = np.full(n, BUCKETS, np.int32)
    masks = np.zeros(n, np.int32)
    for t in range(n):
        gid = token_base + t
        source = gid // per_source
        old = [sum((1 << j) for j in range(k) if owner[t, j] == r) for r in range(8)]
        executable = [sum((1 << j) for j in range(k) if support[r, x[t, j]]) for r in range(8)]
        best = -1
        for edge in range(64):
            a, b = divmod(edge, 8)
            available = old[a] & executable[b]
            if a == b or available == 0:
                continue
            ca, cb, nav = bin(old[a]).count('1'), bin(old[b]).count('1'), bin(available).count('1')
            lo, hi = min(a, b), max(a, b)
            pair = lo * (16 - lo - 1) // 2 + hi - lo - 1
            tie = 2 * (27 - ((pair - gid - 17 * step) % 28)) + int((a < b) == (((gid // 28 + step) & 1) == 0))
            for q, size in enumerate([1, 2, 4, ca]):
                if q < 3 and size > nav:
                    continue
                if q == 3 and (nav != ca or size in [1, 2, 4]):
                    continue
                gain = int(a != source and size == ca) - int(b != source and cb == 0)
                if gain < 0:
                    continue
                size_tie = 3 - ((q - ((gid // 28 + step) & 3) + 4) & 3)
                score = (((gain * 4 + size_tie) * 9 + cb) * 56 + tie) * 64 + 63 - edge
                if score > best:
                    best = score
                    keys[t] = (edge * 2 + gain) * 9 + size
                    selected = 0
                    left = available
                    for _ in range(size):
                        bit = left & -left
                        selected |= bit
                        left ^= bit
                    masks[t] = selected
    return keys, masks


def round_with_counts(x, seed, support, step, global_counts=True):
    r, s, k = x.shape
    shards = [propose(x[i], seed[i], support, s, i * s, step) for i in range(r)]
    counts = np.stack([np.bincount(keys, minlength=BUCKETS + 1)[:BUCKETS] for keys, masks in shards])
    result = seed.copy()
    for rank, (keys, masks) in enumerate(shards):
        global_count = counts.sum(0) if global_counts else counts[rank]
        prefix = counts[:rank].sum(0) if global_counts else np.zeros(BUCKETS, np.int64)
        quota = np.zeros(BUCKETS, np.int64)
        for a in range(8):
            for b in range(a + 1, 8):
                for size in range(1, 9):
                    ab0 = (a * 8 + b) * 18 + size
                    ba0 = (b * 8 + a) * 18 + size
                    x0, x1 = global_count[ab0], global_count[ab0 + 9]
                    y0, y1 = global_count[ba0], global_count[ba0 + 9]
                    n = min(x0 + x1, y0 + y1, max(x1, y1))
                    quota[ab0 + 9], quota[ba0 + 9] = min(n, x1), min(n, y1)
                    quota[ab0], quota[ba0] = n - quota[ab0 + 9], n - quota[ba0 + 9]
        ordinal = np.zeros(BUCKETS, np.int64)
        for t in np.argsort(keys, kind='stable'):
            key = keys[t]
            if key == BUCKETS:
                continue
            if ordinal[key] + prefix[key] < quota[key]:
                destination = (key // 9 // 2) % 8
                for j in range(k):
                    if masks[t] & (1 << j):
                        result[rank, t, j] = destination
            ordinal[key] += 1
    return result


def main():
    data = np.load(Path(__file__).parent / 'inputs/tail.npz')
    cases = [('real_tail19', data['topk'], data['seed'], data['support'])]
    rng = np.random.default_rng(20261026)
    for s in [1, 2, 19, 23]:
        e = 32
        x = np.stack([rng.choice(e, 8, replace=False) for _ in range(8 * s)]).reshape(8, s, 8).astype(np.int32)
        home = np.arange(e) // 4
        seed = home[x].astype(np.int32)
        support = np.arange(8)[:, None] == home[None, :]
        for rank in range(8):
            support[rank, rng.choice(np.flatnonzero(home != rank), 8, replace=False)] = True
        cases.append((f'random_tail{s}', x, seed, support))
    results = []
    for name, x, seed, support in cases:
        resident = np.sum(support.astype(np.int64) * (1 << np.arange(8))[:, None], axis=0)
        for coordinated in [False, True]:
            owner = seed.copy()
            for step in range(4):
                if coordinated:
                    expected = _run_round(x.reshape(-1, 8), owner.reshape(-1, 8), resident, 8,
                                          x.shape[1], 1, False, step, 0)[0].reshape(x.shape)
                else:
                    expected = np.stack([_run_round(x[r], owner[r], resident, 8, x.shape[1],
                                                   1, False, step, r * x.shape[1])[0] for r in range(8)])
                owner = round_with_counts(x, owner, support, step, coordinated)
                assert np.array_equal(owner, expected), (name, coordinated, step, np.count_nonzero(owner != expected))
                assert support[owner, x].all()
                assert np.array_equal(np.bincount(owner.ravel(), minlength=8), np.bincount(seed.ravel(), minlength=8))
            if name == 'real_tail19':
                assert np.array_equal(owner, data['expected_global' if coordinated else 'expected_local'])
            results.append(dict(case=name, coordinated=coordinated, four_rounds_exact=True))
    destination = Path(__file__).parent / 'protocol_audit.json'
    destination.write_text(json.dumps({'scope': 'CPU-only independent proposal and count-prefix protocol check', 'cases': results}, indent=2) + '\n')
    print(json.dumps({'cases': len(results), 'rounds_compared': 4 * len(results), 'all_exact': True}))


if __name__ == '__main__':
    main()
