"""Freeze native seeds, support and CPU outputs for the top-8 GPU component test."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
from moonep_cpu import plan
from vendor.bucket_optimizer import optimize
from replay_strategies import sharded


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--methods', nargs='+', default=['seed', 'visited', 'reciprocal'])
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'inputs')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for layer in a.layers:
        trace = ROOT / f'outputs/trace_audit/layer_{layer:02d}.npz'
        topk = np.load(trace, allow_pickle=False)['topk']
        native = plan(topk, E=352, B=8)
        assert (native.physical_topk >= 0).all()
        seed = native.owner.astype(np.int32)
        for method in a.methods:
            base = ROOT / f'outputs/replay/layer{layer:02d}_b8_{method}'
            global_path = Path(str(base) + '_global_4.npz')
            local_path = Path(str(base) + '_source_sharded_4.npz')
            g = np.load(global_path, allow_pickle=False)
            l = np.load(local_path, allow_pickle=False)
            support = g['support']
            assert np.array_equal(support, l['support'])
            assert support[seed, topk].all()
            target = a.output / f'layer{layer:02d}_{method}.npz'
            np.savez_compressed(target, topk=topk, seed=seed, support=support,
                                expected_global=g['owner'], expected_local=l['owner'])
            records.append({'file': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                            'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
                            'global_reference_sha256': hashlib.sha256(global_path.read_bytes()).hexdigest(),
                            'local_reference_sha256': hashlib.sha256(local_path.read_bytes()).hexdigest(),
                            'shape': list(topk.shape), 'budget': 8, 'layer': layer, 'method': method})
            if layer == a.layers[0] and method == a.methods[-1]:
                # 19 tokens/source gives 152 global and 19 local tokens, exercising
                # both incomplete half-warp partner and incomplete block tails.
                tail_x = np.ascontiguousarray(topk[:, :19])
                tail_seed = np.ascontiguousarray(seed[:, :19])
                tail_g, _ = optimize(tail_x, tail_seed, support, rounds=4, mode='bundle')
                tail_l, _ = sharded(tail_x, tail_seed, support, rounds=4)
                np.savez_compressed(a.output / 'tail.npz', topk=tail_x, seed=tail_seed,
                                    support=support, expected_global=tail_g, expected_local=tail_l)
            print(target.name, flush=True)
    (a.output / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
