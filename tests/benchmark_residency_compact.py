"""Single-GPU old/bitmap/single-package greedy on identical real candidates.

  python -m tests.benchmark_residency_compact --fixture-root /path/to/kernel_inputs \
    --output /fresh/compact_greedy.json

The fixture supplies all eight logical route/seed shards and the exact native
copy list. Untimed source-local gather kernels build the normal sample packet;
normal cap128 selection builds candidates once. The diagnostic replay
graphs then consume the same packet/candidates/active tables. Each timed Graph
contains mandatory initialization + one greedy kernel. No distributed planner,
sample collective, candidate construction, exchange, or payload is benchmarked.
An independent capped CPU oracle and full output/state hashes are checked.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import socket
import statistics
import traceback

import numpy as np
import torch

from moonep.sampled_residency import Residency
from tests.sampled_residency_reference import reference
from tests.validate_sampled_residency_table import SEED, assert_exact, sha, trace_input


ROOT = Path(__file__).resolve().parents[1]


def host(t):
    return t.detach().cpu().numpy().copy()


def digest(value):
    if isinstance(value, np.ndarray):
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return value


def snapshot(planner):
    count = int(planner.history_count.item())
    assert 0 <= count <= planner.history.shape[0]
    return dict(support=host(planner.support), extra=host(planner.extra),
        copies=host(planner.experts_to_copy), history=host(planner.history[:count]),
        history_count=count, status=int(planner.status.item()))


def compare(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        if isinstance(expected[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            assert actual[key] == expected[key], (key, actual[key], expected[key])


def candidate_hashes(planner):
    pc, retained = int(planner.package_count.item()), int(planner.retained_count.item())
    nk, nr = planner.active_counts.cpu().tolist()
    values = {name: host(getattr(planner, name)) for name in (
        'packet', 'masks', 'frequencies', 'classes', 'groups', 'records', 'sizes')}
    values.update(packages=host(planner.packages[:pc]), selected_ids=host(planner.selected_ids[:retained]),
                  active_keys=host(planner.active_keys[:nk]), active_rows=host(planner.active_rows[:nr]))
    return {key: digest(value) for key, value in values.items()}


def graph_timings(graphs, samples, replays):
    """Alternate order across samples, retain every CUDA-event duration."""
    for _ in range(5):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    raw = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for sample in range(samples):
        names = list(graphs)
        shift = sample % len(names)
        order = names[shift:] + names[:shift]
        if (sample // len(names)) % 2:
            order.reverse()
        item = dict(sample=sample, order=order, replay_count=replays, us={})
        for variant in order:
            start.record()
            for _ in range(replays):
                graphs[variant].replay()
            end.record(); end.synchronize()
            item['us'][variant] = 1000*start.elapsed_time(end)/replays
        raw.append(item)
    median = {variant: statistics.median(x['us'][variant] for x in raw) for variant in graphs}
    return dict(raw=raw, median_us=median, compact_over_old=median['compact']/median['old'],
                compact_single_over_compact=median['compact_single']/median['compact'],
                compact_single_over_old=median['compact_single']/median['old'],
                timing_scope='Uninstrumented CUDA Graph: mandatory_validate + one greedy kernel, same candidate buffers.')


def profile_resources(graphs, directory, layer):
    """Profiler resources/per-kernel durations are separate diagnostic evidence."""
    result = {}
    for variant, graph in graphs.items():
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False) as profiler:
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
        path = directory/f'layer{layer:02d}_{variant}.json'
        packed = path.with_suffix('.json.gz')
        if path.exists() or packed.exists():
            raise FileExistsError(path)
        profiler.export_chrome_trace(str(path))
        raw = path.read_bytes()
        events = sorted((x for x in json.loads(raw)['traceEvents']
                         if x.get('cat') == 'kernel' and x.get('ph') == 'X'), key=lambda x: x['ts'])
        assert len(events) == 10, (variant, 'expected five two-kernel diagnostic replays', len(events))
        expected_greedy = ('global_greedy_compact_active<1>' if variant == 'compact_single'
                           else 'global_greedy_compact_active<11>' if variant == 'compact'
                           else 'global_greedy_table_active<')
        for replay in range(5):
            assert 'mandatory_validate(' in events[2*replay]['name']
            assert expected_greedy in events[2*replay+1]['name']
        kernels = []
        for index in (0, 1):
            k = events[index]
            kernels.append(dict(name=k['name'], grid=k['args'].get('grid'), block=k['args'].get('block'),
                registers_per_thread=k['args'].get('registers per thread'),
                shared_memory_bytes=k['args'].get('shared memory'),
                raw_duration_us=[events[2*i+index]['dur'] for i in range(5)],
                # Drop the first two complete profiled replays uniformly.
                retained_median_us=statistics.median(events[2*i+index]['dur'] for i in range(2, 5))))
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        packed.write_bytes(compressed); path.unlink()
        result[variant] = dict(path=str(packed), sha256=sha(packed), raw_sha256=hashlib.sha256(raw).hexdigest(),
            kernels=kernels, discarded_replays=2, retained_replays=3,
            warning='Profiler resources/individual kernel times are diagnostics, not formal complete-planner timing.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--profile-dir', type=Path)
    ap.add_argument('--layers', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--replays', type=int, default=50)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(',')]
    if not layers or len(set(layers)) != len(layers) or any(x not in range(8) for x in layers):
        raise ValueError('layers must be unique in 0..7')
    if min(args.samples, args.replays) < 1:
        raise ValueError('samples/replays must be positive')
    args.profile_dir = args.profile_dir or args.output.with_name(args.output.stem+'_traces')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        file.write('{"status":"initializing"}\n')
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.device); torch.set_num_threads(1)
    device = torch.device('cuda', args.device)
    props = torch.cuda.get_device_properties(device)
    record = dict(status='running', metadata=dict(host=socket.gethostname(), device=args.device,
        gpu=torch.cuda.get_device_name(device), uuid=str(props.uuid), sms=props.multi_processor_count,
        torch=torch.__version__, cuda=torch.version.cuda, logical_ranks=8, physical_gpus=1,
        candidate_limit=128, sm_limit=32, sample_size=256, groups_per_class=2, seed=SEED,
        variants=['old', 'compact', 'compact_single'],
        compact_packages_per_thread=dict(compact=11, compact_single=1),
        source_files={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT/'moonep/sampled_residency.py', ROOT/'moonep/csrc/sampled_residency.cu',
            ROOT/'tests/sampled_residency_reference.py', ROOT/'tests/validate_sampled_residency_table.py']},
        limitation='Component microbenchmark with saved native seed/copies; no eight-GPU complete-planner latency claim.'), layers=[])

    def save():
        args.output.write_text(json.dumps(record, indent=2)+'\n')

    try:
        for layer in layers:
            data = trace_input(args.fixture_root, layer)
            routes, owners, copies = [torch.as_tensor(data[name], device=device) for name in ('routes', 'owner', 'copies')]
            planner = Residency(dict(R=8, E=352, K=8, S=16768, B=8, rank=0, device=device), SEED,
                sample_size=256, groups_per_class=2, candidate_limit=128, component_only=True,
                sm_limit=32, fuse=False)
            assert planner.ext.uses_compact_greedy_table(), 'This configuration lacks compact greedy support'
            np.testing.assert_array_equal(host(planner.ids), data['ids'])
            packet = torch.zeros_like(planner.packet)
            # Untimed component setup: exactly the same gather kernel and
            # integer sum as the distributed sample packet, across all sources.
            for source in range(8):
                planner.gather_local(routes[source], owners[source], copies, source_rank=source)
                packet.add_(planner.packet)
            planner.build_from_reduced_samples(copies, packet)
            torch.cuda.synchronize()
            oracle = reference(data['routes'], data['owner'], data['copies'], data['ids'],
                experts=352, groups_per_class=2, candidate_limit=128)
            cpu_check = assert_exact(planner, oracle, f'layer{layer:02d}/full_selection')
            expected = snapshot(planner)
            before = candidate_hashes(planner)
            row = dict(layer=layer, fixture_sha256=data['fixture_sha256'], cpu_oracle=cpu_check,
                candidates=planner.candidate_diagnostics(), immutable_candidate_hashes=before,
                expected_output_hashes={k: digest(v) for k, v in expected.items()}, checks={})
            record['layers'].append(row); save()
            graphs = {}
            try:
                for variant in ('old', 'compact', 'compact_single'):
                    compact = variant != 'old'
                    single_package = variant == 'compact_single'
                    planner.diagnostic_greedy_replay(copies, compact=compact, single_package=single_package)
                    torch.cuda.synchronize(); compare(expected, snapshot(planner))
                    graph = torch.cuda.CUDAGraph(); graphs[variant] = graph
                    with torch.cuda.graph(graph):
                        planner.diagnostic_greedy_replay(copies, compact=compact, single_package=single_package)
                    graph.replay(); torch.cuda.synchronize()
                    compare(expected, snapshot(planner))
                    row['checks'][variant+'_eager_and_graph_exact'] = True
                row['timing'] = graph_timings(graphs, args.samples, args.replays)
                for variant, graph in graphs.items():
                    graph.replay(); torch.cuda.synchronize()
                    compare(expected, snapshot(planner))
                    row['checks'][variant+'_after_measure_exact'] = True
                assert before == candidate_hashes(planner), 'Diagnostic replay mutated read-only candidate input'
                row['checks']['candidate_inputs_unchanged'] = True
                row['resources'] = profile_resources(graphs, args.profile_dir, layer)
                for variant, graph in graphs.items():
                    graph.replay(); torch.cuda.synchronize()
                    compare(expected, snapshot(planner))
                    row['checks'][variant+'_after_profiler_exact'] = True
                assert before == candidate_hashes(planner)
                row['completed'] = True; save()
                print(json.dumps(dict(layer=layer, **row['timing']['median_us'],
                    ratio=row['timing']['compact_over_old'])), flush=True)
            finally:
                torch.cuda.synchronize()
                for graph in graphs.values():
                    graph.reset()
        record['status'] = 'complete'; save()
    except Exception:
        record['status'] = 'failed'; record['failure'] = traceback.format_exc(); save()
        raise


if __name__ == '__main__':
    main()
