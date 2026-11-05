"""Two-physical-GPU sampled-residency audit, all eight prepared real layers.

Checks exact support, package count and all accepted-history fields against
the saved CPU reference, joint runtime route/owner gathering, final copy list,
and both exchange implementations against each other after every round.
No native MoonEP Buffer is needed; this is a correctness audit, not timing.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
import torch.distributed as dist
from residency_cuda import TwoRankResidency
from exchange_cuda import TwoRankExchange

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=ROOT/'outputs/two_gpu_overlap/inputs')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/two_gpu_overlap/residency_smoke_attempt01.json')
    parser.add_argument('--layers', type=int, nargs='+', default=list(range(8)))
    args = parser.parse_args()
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    assert dist.get_world_size() == 2
    manifest = json.loads((args.input_dir/'manifest.json').read_text())
    expected_by_layer = {r['layer']: r for r in manifest['layer_records']}
    result = {'status': 'running', 'physical_gpus': 2, 'layers': [], 'failures': []}
    try:
        for layer in args.layers:
            error = None
            report = {'layer': layer, 'rank': rank}
            try:
                path = args.input_dir/f'layer_{layer:02d}.npz'
                raw = np.load(path, allow_pickle=False)
                x, old = raw['topk'], raw['owner']
                R, S, K = x.shape
                assert R == 2 and K == 8
                def gpu(a, dtype=torch.int32):
                    return torch.tensor(a, device='cuda', dtype=dtype)
                routes, owner = gpu(x[rank]), gpu(old[rank])
                ids = gpu(raw['sample_ids'], torch.int64)
                residency = TwoRankResidency(gpu(raw['sample_routes']), ids, local_tokens=S, rank=rank)
                native_etc = gpu(raw['native_experts_to_copy'])
                support = residency.run(native_etc, owner, local_routes=routes)
                torch.cuda.synchronize()
                assert np.array_equal(residency.routes.cpu().numpy(), raw['sample_routes']), 'joint route gathering'
                assert np.array_equal(residency.sample_owner.cpu().numpy(), old.reshape(-1, 8)[raw['sample_ids']]), 'joint owner gathering'
                assert np.array_equal(support.cpu().numpy(), raw['support']), 'GPU support != CPU support'
                expected = expected_by_layer[layer]['residency']
                diag = residency.diagnostics.cpu().tolist()
                assert diag == [expected['candidate_packages'], *expected['candidate_extra']], 'GPU diagnostics != CPU'
                count = int(residency.history_count.item())
                hist = residency.history[:count].cpu().tolist()
                expected_history = [[h['a'], h['b'], len(h['add_a']), len(h['add_b']), h['size'], h['estimated_sample_gain'], len(h['add_a'])+len(h['add_b'])]
                                    for h in expected['accepted_packages']]
                assert hist == expected_history, ('GPU history != CPU history', hist, expected_history)
                output_etc = residency.experts_to_copy.cpu().numpy()
                for r in range(2):
                    e = np.flatnonzero(raw['support'][r] & (np.arange(352)//176 != r))
                    want = np.full(8, -1, np.int32);want[:len(e)] = e
                    assert np.array_equal(output_etc[r], want), 'copy list != final support'
                # Repeated GPU select must not retain stale frequencies/history.
                support_again = residency.run(native_etc, owner, local_routes=routes)
                assert np.array_equal(support_again.cpu().numpy(), raw['support'])
                prefix = TwoRankExchange(routes, support, rank=rank, rounds=4, variant='prefix')
                sort = TwoRankExchange(routes, support, rank=rank, rounds=4, variant='sort')
                prefix.reset(owner);sort.reset(owner)
                checks = []
                for step in range(4):
                    prefix.run_round(step);sort.run_round(step)
                    assert torch.equal(prefix.owner, sort.owner), 'prefix/sort owner mismatch'
                    assert torch.equal(prefix.counts, sort.counts), 'prefix/sort counts mismatch'
                    assert torch.equal(prefix.stats[step], sort.stats[step]), 'prefix/sort stats mismatch'
                    checks.append({'round': step + 1, 'exact_owner_counts_stats': True,
                                   'global_stats': prefix.stats[step].cpu().tolist()})
                all_owner = [torch.empty_like(prefix.owner) for _ in range(2)]
                dist.all_gather(all_owner, prefix.owner)
                final = torch.stack(all_owner).cpu().numpy()
                assert np.array_equal(np.bincount(final.ravel(), minlength=2), np.bincount(old.ravel(), minlength=2)), 'load conservation'
                assert np.all(raw['support'][final, x]), 'unsupported final pair'
                remote_before = sum(int(np.any(old[r] != r, axis=1).sum()) for r in range(2))
                remote_after = sum(int(np.any(final[r] != r, axis=1).sum()) for r in range(2))
                predicted = int(prefix.stats[:, 3].sum().item())
                assert remote_before-remote_after == predicted and predicted >= 0, 'predicted vs exact communication'
                report.update(exact_support=True, exact_history=True, exact_candidate_count=True,
                              exact_joint_gather=True, exact_copy_list=True,
                              expected_history=expected_history, diagnostics=diag,
                              exchange_rounds=checks, load_conservation=True,
                              exact_predicted_gain=True, remote_before=remote_before,
                              remote_after=remote_after, final_owner_sha256=hashlib.sha256(final.tobytes()).hexdigest(),
                              input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              residency_sources=residency.source_manifest(), exchange_sources=prefix.source_manifest())
            except BaseException as exc:
                error = {'layer': layer, 'rank': rank, 'error': repr(exc), 'traceback': traceback.format_exc()}
            errors = [None, None]
            dist.all_gather_object(errors, error)
            failures = [e for e in errors if e is not None]
            if failures:
                result['failures'].extend(failures)
                raise RuntimeError('two-rank smoke failed; see retained failure details')
            reports = [None, None]
            dist.all_gather_object(reports, report)
            result['layers'].append({'layer': layer, 'rank_checks': reports})
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print(f'layer {layer}: support/history/copies, four exchange rounds, load and communication passed', flush=True)
        result['status'] = 'complete'
    except BaseException:
        result['status'] = 'failed'
        raise
    finally:
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
