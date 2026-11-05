"""CPU structural oracle and optional GPU exact audit of original R8 residency.

--cpu-only needs NumPy only. GPU execution must be scheduled by the caller.
The oracle reconstructs flattened pair/class/package order independently, then
compares complete accepted-package histories with the freshly replayed CPU
select_support histories in traffic.json. Optional GPU checks every workspace.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAIRS = list(itertools.combinations(range(8), 2))


def reference(routes, owner, etc, ids):
    R, S, K = routes.shape
    assert (R, K) == (8, 8) and len(ids) == 256
    assert np.all(np.diff(np.sort(routes, axis=-1), axis=-1) > 0), "unique top-k required"
    support = np.arange(352)[None, :] // 44 == np.arange(8)[:, None]
    for rank in range(8):
        support[rank, etc[rank][etc[rank] >= 0]] = True
    extra = np.count_nonzero(support & (np.arange(352)[None, :] // 44 != np.arange(8)[:, None]), axis=1)
    initial_support = support.copy()
    flat, own = routes.reshape(-1, 8), owner.reshape(-1, 8)
    classes = np.full((28, 256), -1, np.int32)
    sig0, sig1 = np.zeros((28, 256), np.uint64), np.zeros((28, 256), np.uint64)
    masks = np.zeros((28, 256, 6), np.uint64)
    groups, bitmasks = {}, {}
    for p, (a, b) in enumerate(PAIRS):
        for j, token in enumerate(ids):
            src = int(token // S)
            ga = tuple(sorted(int(e) for e, o in zip(flat[token], own[token]) if o == a))
            gb = tuple(sorted(int(e) for e, o in zip(flat[token], own[token]) if o == b))
            choices = []
            for direction, (old, new, g, reverse) in enumerate(((a, b, ga, gb), (b, a, gb, ga))):
                gain = int(old != src) - int(new != src and not reverse)
                if g and gain >= 0 and sum(not support[new, e] for e in g) <= etc.shape[1] - extra[new]:
                    choices.append((gain, direction, g))
            if not choices:
                continue
            maxgain = max(x[0] for x in choices)
            choices = [x for x in choices if x[0] == maxgain]
            tie = (int(token) + a * 0x9e3779b9 + b * 0x85ebca6b) & 0xffffffff
            tie ^= tie >> 16
            tie = (tie * 0x7feb352d) & 0xffffffff
            tie ^= tie >> 15
            gain, direction, group = choices[tie % len(choices)]
            classes[p, j] = (direction * 9 + len(group)) * 2 + gain
            packed = group + (511,) * (8 - len(group))
            h = 0
            for e in packed[:7]:
                h = (h << 9) | e
            sig0[p, j], sig1[p, j] = h, packed[7]
            rid = p * 256 + j
            groups[rid] = group
            bitmasks[rid] = sum(1 << e for e in group)
            for e in group:
                masks[p, j, e // 64] |= np.uint64(1) << np.uint64(e % 64)
    freq = np.zeros((28, 256), np.int32)
    tops, first = np.full((28, 36, 2), -1, np.int32), np.full((28, 36), 256, np.int32)
    for p in range(28):
        occurrences = {}
        for j in range(256):
            if classes[p, j] >= 0:
                occurrences.setdefault((int(classes[p, j]), groups[p * 256 + j]), []).append(j)
        for indices in occurrences.values():
            freq[p, indices[0]] = len(indices)
        for c in range(36):
            matches = np.flatnonzero(classes[p] == c)
            if len(matches):
                first[p, c] = matches[0]
            leaders = [p * 256 + int(j) for j in matches if freq[p, j] > 0]
            leaders.sort(key=lambda rid: (-int(freq.ravel()[rid]), groups[rid]))
            tops[p, c, :min(2, len(leaders))] = leaders[:2]
    records, sizes = np.full((576, 4), -1, np.int32), np.zeros(576, np.int32)
    packages = []
    for p, (a, b) in enumerate(PAIRS):
        for size in range(1, 9):
            ka, kb = (a * 8 + b) * 9 + size, (b * 8 + a) * 9 + size
            for direction, key in ((0, ka), (1, kb)):
                gains = sorted((0, 1), key=lambda gain: int(first[p, (direction * 9 + size) * 2 + gain]))
                ids_in_bucket = [int(i) for gain in gains for i in tops[p, (direction * 9 + size) * 2 + gain] if i >= 0]
                sizes[key] = len(ids_in_bucket)
                records[key, :len(ids_in_bucket)] = ids_in_bucket
            for ra in records[ka, :sizes[ka]]:
                for rb in records[kb, :sizes[kb]]:
                    if (classes.ravel()[ra] & 1) + (classes.ravel()[rb] & 1):
                        packages.append((a, b, size, int(ra), int(rb)))
    resident = [sum(1 << int(e) for e in np.flatnonzero(row)) for row in support]
    history, history_rows = [], []
    def enabled_counts(key, bits):
        n = [0, 0]
        for rec in records[key, :sizes[key]]:
            if bitmasks[rec] & ~bits == 0:
                n[classes.ravel()[rec] & 1] += int(freq.ravel()[rec])
        return n
    def match(x, y):
        n = min(sum(x), sum(y), max(x[1], y[1]))
        return min(n, x[1]) + min(n, y[1])
    while True:
        enabled = [enabled_counts(key, resident[(key // 9) % 8]) for key in range(576)]
        best = None
        for p, (a, b, size, ra, rb) in enumerate(packages):
            aa, ab = bitmasks[rb] & ~resident[a], bitmasks[ra] & ~resident[b]
            ca, cb = bin(aa).count("1"), bin(ab).count("1")
            cost = ca + cb
            if cost == 0 or extra[a] + ca > etc.shape[1] or extra[b] + cb > etc.shape[1]:
                continue
            ka, kb = (a * 8 + b) * 9 + size, (b * 8 + a) * 9 + size
            gain = match(enabled_counts(ka, resident[b] | ab), enabled_counts(kb, resident[a] | aa)) - match(enabled[ka], enabled[kb])
            if gain <= 0:
                continue
            score = (gain / cost, gain, -cost, -p)
            if best is None or score > best[0]:
                best = score, p, a, b, size, gain, aa, ab, ca, cb
        if best is None:
            break
        _, p, a, b, size, gain, aa, ab, ca, cb = best
        resident[a] |= aa
        resident[b] |= ab
        extra[a] += ca
        extra[b] += cb
        additions_a = [e for e in range(352) if (aa >> e) & 1]
        additions_b = [e for e in range(352) if (ab >> e) & 1]
        history.append(dict(a=a, b=b, add_a=additions_a, add_b=additions_b, size=size, estimated_sample_gain=gain))
        history_rows.append([p, gain, a, b, size, ca + cb] + [(bits >> (64*w)) & ((1 << 64) - 1) for bits in (aa, ab) for w in range(6)])
    result_support = np.array([[(bits >> e) & 1 for e in range(352)] for bits in resident], dtype=bool)
    return dict(classes=classes, sig0=sig0, sig1=sig1, masks=masks, frequencies=freq,
        tops=tops, first=first, records=records, sizes=sizes, packages=np.asarray(packages, np.int32).reshape(-1, 5),
        support=result_support, extra=extra, history=history, history_rows=history_rows,
        initial_support=initial_support)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    traffic = json.loads((args.root / "outputs/eight_rank_estimate/traffic.json").read_text())
    ids = np.asarray(traffic["sampling"]["sample_ids"], np.int64)
    output = dict(status="running", scope="CPU flattened-order oracle" if args.cpu_only else "GPU exact intermediate/support/history and repeat/Graph audit; no timing", layers=[], failures=[])
    if not args.cpu_only:
        import torch
        from residency8 import Residency8
        device = "cuda"
        output["gpu"] = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layer in map(int, args.layers.split(",")):
        with np.load(args.root / f"outputs/trace_audit/layer_{layer:02}.npz") as inp:
            routes = inp["topk"]
        with np.load(args.root / f"outputs/rt2_b8_comparison/layer{layer:02}_native_rt0.npz") as inp:
            owner, etc = inp["owner"].astype(np.int32), inp["experts_to_copy"]
        ref = reference(routes, owner, etc, ids)
        original = traffic["layers"][layer]["methods"]["sample256_g2_bundle"]
        assert ref["history"] == original["residency_diagnostics"]["accepted_packages"], (layer, "CPU history")
        with np.load(args.root / f"outputs/replay/layer{layer:02}_b8_small256_global_4.npz") as inp:
            assert np.array_equal(ref["support"], inp["support"]), (layer, "CPU support")
        row = dict(layer=layer, cpu_history_exact=True, cpu_support_exact=True, packages=len(ref["packages"]), accepted=len(ref["history"]))
        if not args.cpu_only:
            gpu = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=device)
            planner = Residency8(gpu(routes), gpu(ids), per_source=routes.shape[1])
            devowner, devetc = gpu(owner), gpu(etc)
            for repeat in range(2):
                planner.run(devowner, devetc)
                torch.cuda.synchronize()
                for key in ("classes", "sig0", "sig1", "masks", "frequencies", "tops", "first", "records", "sizes", "extra", "support"):
                    actual = getattr(planner, key).cpu().numpy()
                    expected = ref[key]
                    if expected.dtype == np.uint64:
                        actual = actual.view(np.uint64)
                    assert np.array_equal(actual, expected), (layer, repeat, key, np.count_nonzero(actual != expected))
                count = int(planner.package_count.item())
                assert count == len(ref["packages"])
                assert np.array_equal(planner.packages[:count].cpu().numpy(), ref["packages"])
                assert planner.accepted_packages() == ref["history"], (layer, repeat, "GPU history")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                planner.run(devowner, devetc)
            graph.replay()
            torch.cuda.synchronize()
            assert np.array_equal(planner.support.cpu().numpy(), ref["support"])
            assert planner.accepted_packages() == ref["history"]
            graph.reset()
            row.update(gpu_all_intermediates_exact=True, gpu_history_exact=True, repeated_reset_exact=True, graph_exact=True)
            output["source_sha256"] = planner.source_manifest()
            del graph, planner, devowner, devetc
        output["layers"].append(row)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    output["status"] = "complete"
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
