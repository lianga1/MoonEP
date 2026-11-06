"""Pure-host parser for the SM32 stage profiler; never infers unobserved waits.

The eager trace maps each individual GPU launch to a labeled Python phase.
The steady CUDA Graph trace must repeat that exact launch sequence. The first
two complete executions are retained as raw evidence but excluded from timing.
"""
from collections import defaultdict
import gzip
import json
import math
import statistics


def load_trace(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as file:
        return json.load(file)


def kernels(data):
    return sorted((e for e in data['traceEvents'] if e.get('cat') == 'kernel'
                   and e.get('ph') == 'X'), key=lambda e: e['ts'])


def attribution(name, stage):
    """Semantic scope of elapsed kernel time, not an isolated hardware cost."""
    if stage == 'native_seed':
        return 'native_compute_peer_io_and_wait_mixed'
    if stage in ('pre_sync', 'publication_barrier_before', 'publication_barrier_after'):
        return 'peer_barrier_signal_and_wait'
    if 'collective' in stage:
        if 'barrier_kernel(' in name:
            return 'peer_barrier_signal_and_wait'
        if 'read_kernel(' in name:
            return 'peer_payload_read_and_reduce'
        if 'copy_kernel(' in name:
            return 'local_payload_copy_and_system_fence'
        if 'fused_collective' in name:
            return 'peer_copy_read_grid_sync_and_wait_mixed'
        raise AssertionError(('unclassified collective kernel', name, stage))
    if stage == 'publication_write':
        return 'peer_metadata_write'
    return 'local_kernel_work'


def eager_mapping(data, stage_manifest):
    stages = {x['name']: x for x in stage_manifest}
    labels = [e for e in data['traceEvents'] if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('sm32_stage/')]
    assert len(labels) == len(stages), ('missing/duplicate eager stage labels', len(labels), len(stages))
    labels_by_name = {e['name'].split('/', 2)[2]: e for e in labels}
    assert set(labels_by_name) == set(stages)
    launches = defaultdict(list)
    for event in data['traceEvents']:
        if event.get('cat') in ('cuda_runtime', 'cuda_driver') and event.get('ph') == 'X':
            corr = event.get('args', {}).get('correlation')
            if corr is not None:
                launches[corr].append(event)
    mapped = []
    for ordinal, kernel in enumerate(kernels(data)):
        corr = kernel.get('args', {}).get('correlation')
        matches = []
        for launch in launches[corr]:
            for stage, label in labels_by_name.items():
                if (label['pid'] == launch['pid'] and label['tid'] == launch['tid']
                        and label['ts'] <= launch['ts'] < label['ts'] + label['dur']):
                    matches.append(stage)
        assert len(set(matches)) == 1, ('kernel stage mapping is ambiguous', kernel['name'], corr, matches)
        stage = matches[0]
        mapped.append(dict(ordinal=ordinal, stage=stage, name=kernel['name'],
                           category=attribution(kernel['name'], stage)))
    counts = {name: sum(x['stage'] == name for x in mapped) for name in stages}
    assert all(counts.values()), ('stage without a kernel', counts)
    assert [x['stage'] for x in mapped] == sorted(
        [x['stage'] for x in mapped], key=lambda name: stages[name]['index'])
    return dict(launches=mapped, stage_kernel_counts=counts,
                note='Eager durations are intentionally unused; only correlation/ordered names map stages.')


def summarize_graph(data, mapping, discard, kept, sm_limit):
    expected = mapping['launches']
    all_kernels = kernels(data)
    n = len(expected)
    assert n and len(all_kernels) == n * (discard + kept), (
        'incomplete or extra Graph kernels', len(all_kernels), n, discard, kept)
    raw = []
    for replay in range(discard + kept):
        chunk = all_kernels[replay*n:(replay+1)*n]
        assert [x['name'] for x in chunk] == [x['name'] for x in expected], (
            'Graph launch sequence differs from eager', replay,
            [(i, x['name'], y['name']) for i, (x, y) in enumerate(zip(chunk, expected))
             if x['name'] != y['name']])
        streams = sorted(set(x['args']['stream'] for x in chunk))
        assert len(streams) == 1, ('planner no longer executes on one stream', streams)
        phase = defaultdict(float)
        category = defaultdict(float)
        launches = []
        begin = chunk[0]['ts']
        last_end = begin
        positive_gaps = 0.0
        for event, item in zip(chunk, expected):
            grid = event['args'].get('grid')
            assert grid is not None, ('kernel has no grid metadata', event['name'])
            blocks = math.prod(grid)
            assert blocks <= sm_limit, ('SM-budget exceeded', event['name'], blocks, sm_limit)
            assert 'nccl' not in event['name'].lower()
            gap = event['ts'] - last_end
            assert gap > -0.1, ('overlapping kernels on one stream', gap)
            positive_gaps += max(0., gap)
            last_end = event['ts'] + event['dur']
            phase[item['stage']] += event['dur']
            category[item['category']] += event['dur']
            launches.append(dict(**item, start_us=event['ts'], duration_us=event['dur'],
                gap_before_us=gap, blocks=blocks, grid=grid,
                block=event['args'].get('block'), stream=event['args']['stream'],
                registers_per_thread=event['args'].get('registers per thread'),
                shared_memory_bytes=event['args'].get('shared memory')))
        sum_kernel = sum(x['dur'] for x in chunk)
        span = last_end-begin
        raw.append(dict(replay=replay, discarded=replay < discard,
            kernel_sum_us=sum_kernel, first_start_to_last_end_us=span,
            positive_inter_kernel_gaps_us=positive_gaps,
            closure_error_us=span-sum_kernel-positive_gaps,
            stage_kernel_us=dict(phase), category_kernel_us=dict(category), launches=launches))
    retained = raw[discard:]
    median = lambda key: statistics.median(x[key] for x in retained)
    per_kernel = [dict(**item, median_us=statistics.median(x['launches'][i]['duration_us'] for x in retained),
                      raw_us=[x['launches'][i]['duration_us'] for x in retained])
                  for i, item in enumerate(expected)]
    return dict(discarded_replays=discard, retained_replays=kept, raw_replays=raw,
        kernel_count_per_replay=n, per_kernel=per_kernel,
        median_kernel_sum_us=median('kernel_sum_us'),
        median_gpu_span_us=median('first_start_to_last_end_us'),
        median_inter_kernel_gaps_us=median('positive_inter_kernel_gaps_us'),
        stage_median_us={key: statistics.median(x['stage_kernel_us'][key] for x in retained)
                         for key in retained[0]['stage_kernel_us']},
        category_median_us={key: statistics.median(x['category_kernel_us'][key] for x in retained)
                            for key in retained[0]['category_kernel_us']},
        warning='Profiler kernel wall durations include stalls. Barrier time includes signaling, fences and rank arrival skew; fused peer kernels and native seed mix compute/communication/wait. No isolated network-bandwidth or pure-wait claim.')
