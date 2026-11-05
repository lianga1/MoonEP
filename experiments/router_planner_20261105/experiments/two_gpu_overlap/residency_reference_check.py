"""CPU emulation of the GPU exact signatures and bitset residency decisions.

This checks the specialization against saved, independently generated CPU
select_support results. It does not validate CUDA execution or performance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def emulate(routes, ids, owner, native_etc, S, B=8):
    resident = [sum(1 << e for e in range(r*176, (r+1)*176)) for r in range(2)]
    for r in range(2):
        for e in native_etc[r]:
            if e >= 0:
                resident[r] |= 1 << int(e)
    extras = [bin(resident[r]).count('1') - 176 for r in range(2)]
    classes, signatures, groups = [], [], []
    for token, x, own in zip(ids, routes, owner):
        source = int(token)//S
        gs = [sorted(int(e) for e, a in zip(x, own) if a == r) for r in range(2)]
        choices = []
        for a in range(2):
            b = 1-a
            if not gs[a]:
                continue
            v = int(a != source)-int(b != source and not gs[b])
            missing = sum(not ((resident[b] >> e) & 1) for e in gs[a])
            if v >= 0 and missing <= B-extras[b]:
                choices.append((v, a))
        if not choices:
            classes.append(-1);signatures.append((0, 0));groups.append(0)
            continue
        best = max(v for v, _ in choices)
        choices = [(v, a) for v, a in choices if v == best]
        tie = (int(token)+0x85ebca6b)&0xffffffff
        tie ^= tie >> 16;tie = (tie*0x7feb352d)&0xffffffff;tie ^= tie >> 15
        v, a = choices[tie % len(choices)]
        g = gs[a];padded = g+[511]*(8-len(g));lo = 0
        for e in padded[:7]:
            lo = (lo << 9) | e
        classes.append((a*2+v)*9+len(g));signatures.append((lo, padded[7]));groups.append(sum(1 << e for e in g))
    frequencies = []
    for i, c in enumerate(classes):
        matches = [j for j in range(len(classes)) if c >= 0 and c == classes[j] and signatures[i] == signatures[j]]
        frequencies.append(len(matches) if matches and matches[0] == i else 0)
    tops, first = [], []
    for c in range(36):
        matching = [i for i, cl in enumerate(classes) if cl == c]
        first.append(min(matching) if matching else len(classes))
        ranked = sorted([i for i in matching if frequencies[i]], key=lambda i: (-frequencies[i], signatures[i]))[:2]
        tops.append(ranked)
    records = {}
    for direction in range(2):
        for size in range(1, 9):
            order = sorted((0, 1), key=lambda v: first[(direction*2+v)*9+size])
            records[direction, size] = [i for v in order for i in tops[(direction*2+v)*9+size]]
    packages = [(a, b, size) for size in range(1, 9) for a in records[0, size] for b in records[1, size]
                if (classes[a]//9)%2+(classes[b]//9)%2 > 0]
    def enabled_sum(direction, size, bits):
        n = [0, 0]
        for i in records[direction, size]:
            if groups[i] & ~bits == 0:
                n[(classes[i]//9)%2] += frequencies[i]
        return n
    def gain(x, y):
        n = min(sum(x), sum(y), max(x[1], y[1]))
        return min(n, x[1])+min(n, y[1])
    history = []
    while True:
        best = None
        for idx, (a, b, size) in enumerate(packages):
            add_a, add_b = groups[b] & ~resident[0], groups[a] & ~resident[1]
            ca, cb = bin(add_a).count('1'), bin(add_b).count('1')
            cost = ca+cb
            if not cost or extras[0]+ca > B or extras[1]+cb > B:
                continue
            old = gain(enabled_sum(0, size, resident[1]), enabled_sum(1, size, resident[0]))
            delta = gain(enabled_sum(0, size, resident[1] | add_b), enabled_sum(1, size, resident[0] | add_a))-old
            if delta <= 0:
                continue
            score = (delta/cost, delta, -cost, -idx)
            if best is None or score > best[0]:
                best = (score, add_a, add_b, ca, cb, size, delta, cost)
        if best is None:
            break
        _, aa, ab, ca, cb, size, delta, cost = best
        resident[0] |= aa;resident[1] |= ab;extras[0] += ca;extras[1] += cb
        history.append([0, 1, ca, cb, size, delta, cost])
    support = np.array([[bool((bits >> e) & 1) for e in range(352)] for bits in resident])
    return support, history, [len(packages), *extras], sum(c >= 0 and (c//9)%2 == 0 for c in classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=ROOT/'outputs/two_gpu_overlap/inputs')
    args = parser.parse_args()
    manifest = json.loads((args.input_dir/'manifest.json').read_text())
    results = []
    for record in manifest['layer_records']:
        layer = record['layer']
        raw = np.load(args.input_dir/f'layer_{layer:02}.npz', allow_pickle=False)
        support, history, diag, neutral = emulate(raw['sample_routes'], raw['sample_ids'], raw['owner'].reshape(-1, 8)[raw['sample_ids']], raw['native_experts_to_copy'], raw['topk'].shape[1])
        expected = record['residency']
        hist = [[h['a'], h['b'], len(h['add_a']), len(h['add_b']), h['size'], h['estimated_sample_gain'], len(h['add_a'])+len(h['add_b'])] for h in expected['accepted_packages']]
        assert np.array_equal(support, raw['support']), layer
        assert history == hist, (layer, history, hist)
        assert diag == [expected['candidate_packages'], *expected['candidate_extra']], (layer, diag)
        results.append({'layer': layer, 'exact_support_history_diagnostics': True, 'gain_zero_samples': neutral, 'diagnostics': diag})
    print(json.dumps({'cpu_semantics_only': True, 'layers': results}, indent=2))


if __name__ == '__main__':
    main()
