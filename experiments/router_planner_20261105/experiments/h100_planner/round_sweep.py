"""Bundle8 L=1/2/4 sweep on one GPU with the existing H100 benchmark protocol.

Fixed residency and saved rt0 seed are identical across L.  A four-round CPU
execution supplies exact prefix references; correctness checks are not timed.
An existing output path is rejected so previous attempts cannot be overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from baseline_virtual import VirtualMoonEPPlanner
from bundle_lane8 import Bundle8LanePlanner
from materialize import Materializer
from materialize_cuda import CudaMaterializer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    queries = [
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,power.draw", "--format=csv,noheader"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
    ]
    result = []
    for command in queries:
        try:
            p = subprocess.run(command, text=True, capture_output=True, timeout=15)
            result.append(dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
        except Exception as exc:
            result.append(dict(command=command, error=repr(exc)))
    return result


def audit(routes, owner, support, base_remote):
    ranks, per_source, topk = routes.shape
    assert owner.shape == routes.shape
    assert ((owner >= 0) & (owner < ranks)).all()
    assert support[owner, routes].all(), "Unsupported expert execution"
    loads = np.bincount(owner.ravel(), minlength=ranks)
    assert np.array_equal(loads, np.full(ranks, per_source * topk)), "Load changed"
    ordered = np.sort(owner, axis=-1)
    first = np.concatenate((np.ones((ranks, per_source, 1), bool), ordered[:, :, 1:] != ordered[:, :, :-1]), axis=2)
    remote = int((first & (ordered != np.arange(ranks)[:, None, None])).sum())
    return dict(remote=remote, reduction_pct=100 * (1 - remote / base_remote),
                audit_pass=True, loads=loads.tolist(), all_pairs_executable=True,
                one_owner_per_input_pair=True)


def cpu_prefixes(routes, seed, support, base_remote):
    from vendor.bucket_optimizer import HAVE_NUMBA, _run_round
    if not HAVE_NUMBA:
        raise RuntimeError("Numba is required for the full-trace CPU prefix oracle")
    ranks, per_source, topk = routes.shape
    experts = np.ascontiguousarray(routes.reshape(-1, topk), dtype=np.int32)
    owner = np.ascontiguousarray(seed.reshape(-1, topk), dtype=np.int32).copy()
    resident = np.zeros(support.shape[1], dtype=np.int64)
    for rank in range(ranks):
        resident |= support[rank].astype(np.int64) << rank
    expected = {0: seed.copy()}
    records = []
    previous = base_remote
    for step in range(4):
        result = _run_round(experts, owner, resident, ranks, per_source, 1, False, step)
        (owner, proposed, gain_one, checks, candidates, legal, exchanges,
         neutral, moved, accepted, predicted, candidate_sizes,
         proposal_sizes, accepted_sizes) = result
        expected[step + 1] = owner.reshape(routes.shape).copy()
        quality = audit(routes, expected[step + 1], support, base_remote)
        assert previous - quality['remote'] == predicted >= 0
        assert neutral == 0
        records.append(dict(round=step + 1, quality=quality,
                            proposed_tokens=int(proposed), proposed_gain_one=int(gain_one),
                            target_checks=int(checks), candidate_gain_checks=int(candidates),
                            legal_candidates=int(legal), accepted_exchanges=int(exchanges),
                            neutral_exchanges=int(neutral), moved_pairs=int(moved),
                            accepted_tokens=int(accepted), communication_reduction=int(predicted),
                            candidate_sizes=candidate_sizes.tolist(),
                            proposal_sizes=proposal_sizes.tolist(),
                            accepted_exchanges_by_size=accepted_sizes.tolist()))
        previous = quality['remote']
    return expected, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--layers', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=10)
    parser.add_argument('--graph-repeats', type=int, default=20)
    args = parser.parse_args()
    if args.groups < 1 or args.graph_repeats < 1:
        raise ValueError('Repeat counts must be positive')
    root = args.root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write('{"status":"initializing"}\n')
    sys.path.insert(0, str(root / 'analysis'))
    from moonep_cpu import communication

    source_paths = [Path(__file__).resolve(), root / 'analysis/vendor/bucket_optimizer.py',
                    root / 'analysis/moonep_cpu.py']
    for stem in ('bundle_lane8', 'baseline_virtual', 'materialize', 'materialize_cuda'):
        source_paths.extend(p for p in (root / 'experiments/h100_planner').glob(stem + '.*') if p.is_file())
    out = dict(status='running', metadata=dict(
        started_unix=time.time(), platform=platform.platform(), torch=torch.__version__,
        cuda=torch.version.cuda, physical_gpus_used=1, logical_ranks=8,
        rounds=[1, 2, 4], reference_rounds=0, requested_layers=list(map(int, args.layers.split(','))),
        groups=args.groups, graph_repeats=args.graph_repeats,
        protocol='10 eager samples and 20 CUDA Graph samples by default; rotate method order each repetition, exactly as benchmark.py. One eager warmup, three pre-capture warmups. Compile, allocation, correctness checks and graph capture excluded.',
        scope='bundle8_L*: owner reset plus L exchange rounds. bundle8_L*_seed_exchange_output: semantic-port seed generation plus owner reset plus L exchanges plus CUDA materialization. Fixed residency selection, resident-mask packing, input statistics, actual token/weight communication and expert computation excluded.',
        quality='Deduplicated remote (token,destination) count; reduction relative to same saved rt0 seed. Every L uses an exact prefix of the same four-round algorithm.',
        code_sha256={str(p.relative_to(root)): sha(p) for p in dict.fromkeys(source_paths)},
        input_sha256={}, environment_before=environment(),
        env={name: os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES', 'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS')}),
        layers=[], failures=[])

    def save():
        args.output.write_text(json.dumps(out, indent=2))

    def fail(layer, variant, phase, exc):
        out['failures'].append(dict(layer=layer, variant=variant, phase=phase,
                                    error=repr(exc), traceback=traceback.format_exc()))
        save()

    try:
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        out['metadata']['gpu'] = torch.cuda.get_device_name(0)
        out['metadata']['device_properties'] = str(torch.cuda.get_device_properties(0))
        for layer in map(int, args.layers.split(',')):
            paths = [root / f'outputs/trace_audit/layer_{layer:02d}.npz',
                     root / f'outputs/replay/layer{layer:02d}_b8_small256_global_4.npz',
                     root / f'outputs/rt2_b8_comparison/layer{layer:02d}_native_rt0.npz']
            out['metadata']['input_sha256'].update({str(p.relative_to(root)): sha(p) for p in paths})
            with np.load(paths[0]) as data:
                x = data['topk'].astype(np.int32)
            with np.load(paths[1]) as data:
                support = data['support'].copy()
                saved_final = data['owner'].copy()
            with np.load(paths[2]) as data:
                seed = data['owner'].astype(np.int32)
            base_remote = communication(x, seed, 352)['remote_token_destinations']
            record = dict(layer=layer, shape=list(x.shape), base_remote=int(base_remote),
                          qualities={'0': audit(x, seed, support, base_remote)},
                          correctness={}, scope_timings_ms={}, host_timings_ms={},
                          graph_timings_ms={}, graph_audit={})
            out['layers'].append(record)
            save()
            cpu_start = time.perf_counter()
            expected, rounds = cpu_prefixes(x, seed, support, base_remote)
            record['cpu_oracle_seconds_not_gpu_timing'] = time.perf_counter() - cpu_start
            record['cpu_rounds'] = rounds
            assert np.array_equal(expected[4], saved_final), 'CPU L4 differs from saved reference'
            xt = torch.tensor(x, device='cuda')
            ot = torch.tensor(seed, device='cuda')
            st = torch.tensor(support, device='cuda')
            baseline = VirtualMoonEPPlanner(xt)
            baseline.run()
            torch.cuda.synchronize()
            assert np.array_equal(baseline.owner.cpu().numpy().reshape(x.shape), seed)
            baseline.run_seed()
            torch.cuda.synchronize()
            assert np.array_equal(baseline.owner.cpu().numpy().reshape(x.shape), seed)
            mat, oracle = CudaMaterializer(xt, st), Materializer(xt, st)
            runners, functions, targets = {}, {}, {}

            def material_audit(result, target):
                want = oracle.run(torch.as_tensor(target.astype(np.int32), device='cuda'))
                assert result.keys() == want.keys()
                for key in want:
                    assert torch.equal(result[key], want[key]), 'Materializer field ' + key
                raw = result['raw_dst'].cpu().numpy()
                phys = result['physical_topk'].cpu().numpy()
                assert len(np.unique(raw)) == x.size
                assert phys.min() >= 0 and phys.max() < 8 * 52
                provenance = result['provenance'][result['raw_dst'].long()].cpu().numpy()
                wanted = (np.arange(x.size) // mat.cap) * mat.NvS + np.arange(x.size) % mat.cap
                assert np.array_equal(provenance, wanted)

            functions['seed_only'] = baseline.run_seed
            functions['materialize_only'] = lambda: mat.run(ot)
            targets['seed_only'] = targets['materialize_only'] = seed
            for count in (1, 2, 4):
                name = f'bundle8_L{count}'
                runner = Bundle8LanePlanner(xt, st, x.shape[1], rounds=count)
                runners[count] = runner
                result = runner.run(ot)
                torch.cuda.synchronize()
                cpu = result.cpu().numpy().reshape(x.shape)
                assert np.array_equal(cpu, expected[count]), f'CPU mismatch L={count}'
                quality = audit(x, cpu, support, base_remote)
                record['qualities'][str(count)] = quality
                gpu_stats = runner.stats.cpu().numpy()
                expected_stats = np.array([[r['accepted_exchanges'], r['moved_pairs'],
                                            r['accepted_tokens'], r['communication_reduction']]
                                           for r in rounds[:count]], dtype=np.int64)
                assert np.array_equal(gpu_stats, expected_stats), f'Round statistics mismatch L={count}'
                record['correctness'][str(count)] = dict(exact_cpu_prefix=True,
                    exact_saved_l4=(True if count == 4 else None), gpu_round_stats=gpu_stats.tolist(),
                    gpu_stats_fields=['accepted_exchanges', 'moved_pairs', 'accepted_tokens', 'communication_reduction'])
                material_audit(mat.run(result), expected[count])
                full = lambda r=runner: mat.run(r.run(baseline.run_seed().reshape(x.shape)))
                material_audit(full(), expected[count])
                record['correctness'][str(count)]['materialized_unique_provenance_exact'] = True
                functions[name] = lambda r=runner: r.run(ot)
                functions[name + '_seed_exchange_output'] = full
                targets[name] = targets[name + '_seed_exchange_output'] = expected[count]
            # Standalone L1/L2 agree with prefixes of the SAME GPU L4 execution.
            prefix_runner = runners[4]
            prefix_runner.reset(ot.reshape(-1, 8))
            for step in range(4):
                prefix_runner.run_round(step)
                torch.cuda.synchronize()
                assert np.array_equal(prefix_runner.owner.cpu().numpy().reshape(x.shape), expected[step + 1])
            record['gpu_four_round_prefixes_exact_cpu'] = True
            save()

            for function in functions.values():
                function()
            torch.cuda.synchronize()
            names = list(functions)
            record['scope_timings_ms'] = {name: [] for name in names}
            record['host_timings_ms'] = {name: [] for name in names}
            for rep in range(args.groups):
                shift = rep % len(names)
                for name in names[shift:] + names[:shift]:
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter()
                    start.record()
                    functions[name]()
                    end.record()
                    end.synchronize()
                    record['host_timings_ms'][name].append((time.perf_counter() - wall) * 1000)
                    record['scope_timings_ms'][name].append(start.elapsed_time(end))
            save()

            def graph_audit(name, result):
                if isinstance(result, dict):
                    material_audit(result, targets[name])
                else:
                    assert np.array_equal(result.cpu().numpy().reshape(x.shape), targets[name]), name

            graphs, returns = {}, {}
            for name, function in functions.items():
                phase = 'graph_capture'
                try:
                    for _ in range(3):
                        function()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        result = function()
                    graph.replay()
                    torch.cuda.synchronize()
                    phase = 'graph_validation'
                    graph_audit(name, result)
                    graphs[name], returns[name] = graph, result
                    record['graph_timings_ms'][name] = []
                    record['graph_audit'][name] = True
                except Exception as exc:
                    record['graph_audit'][name] = False
                    fail(layer, name, phase, exc)
            names = list(graphs)
            for rep in range(args.graph_repeats):
                shift = rep % len(names)
                for name in names[shift:] + names[:shift]:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    record['graph_timings_ms'][name].append(start.elapsed_time(end))
            for name, graph in graphs.items():
                try:
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_audit(name, returns[name])
                except Exception as exc:
                    record['graph_audit'][name] = False
                    fail(layer, name, 'graph_post_timing_validation', exc)
            record['eager_medians_ms'] = {name: float(np.median(values)) for name, values in record['scope_timings_ms'].items()}
            record['graph_medians_ms'] = {name: float(np.median(values)) for name, values in record['graph_timings_ms'].items() if values}
            save()
            print(json.dumps(dict(layer=layer, quality=record['qualities'], graph_medians_ms=record['graph_medians_ms'])), flush=True)
            # Drop graph captures and runners before the next layer. Tensor
            # references in function closures are replaced on the next loop.
            graphs.clear()
            returns.clear()
            functions.clear()
            runners.clear()
        out['status'] = 'complete' if not out['failures'] else 'complete_with_failures'
        if not out['failures']:
            base_sum = sum(row['base_remote'] for row in out['layers'])
            out['summary'] = dict(
                layer_count=len(out['layers']),
                quality_aggregation='sum remote counts over requested layers; same total rt0 denominator',
                latency_aggregation='arithmetic mean of per-layer medians, matching benchmark.py report',
                total_base_remote=base_sum,
                rounds={str(count): dict(
                    total_remote=sum(row['qualities'][str(count)]['remote'] for row in out['layers']),
                    reduction_pct=100 * (1 - sum(row['qualities'][str(count)]['remote'] for row in out['layers']) / base_sum)
                ) for count in (0, 1, 2, 4)},
                mean_layer_eager_medians_ms={name: float(np.mean([row['eager_medians_ms'][name] for row in out['layers']])) for name in out['layers'][0]['eager_medians_ms']},
                mean_layer_graph_medians_ms={name: float(np.mean([row['graph_medians_ms'][name] for row in out['layers']])) for name in out['layers'][0]['graph_medians_ms']})
    except Exception as exc:
        out['status'] = 'failed'
        fail(locals().get('layer'), 'round_sweep', 'fatal', exc)
        raise
    finally:
        out['metadata']['finished_unix'] = time.time()
        out['metadata']['environment_after'] = environment()
        save()


if __name__ == '__main__':
    main()
