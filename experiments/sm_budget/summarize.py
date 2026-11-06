#!/usr/bin/env python3
"""Independently validate the frozen 03 cohort, raw traces, and timing arithmetic.

Standard library only; no GPU, PyTorch, private router fixtures, or online access.
Missing, inconsistent, or incomplete formal evidence is a hard error.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = {
    'baseline_before_03': (0, False),
    'bounded32_03': (32, False),
    'fused32_03': (32, True),
    'fused48_03': (48, True),
    'fused64_03': (64, True),
    'fused96_03': (96, True),
    'fused128_03': (128, True),
    'baseline_after_03': (0, False),
}
FIELDS = {'dst', 'experts_to_copy', 'zero_fill_ranges', 'remote_stats',
          'cu', 'owner', 'support', 'source_info', 'history'}
ANCHORS = dict(sample_gather=3, expert_greedy=3, exchange_proposal=12,
               exchange_prefix=12, layout_encode=3, publication=3)
CONFIG = dict(sample_size=256, groups_per_class=2, rounds=4,
              sample_seed=20261026, candidate_limit=128)
SHAPE = dict(R=8, N=134144, S=16768, E=352, K=8, B=8)


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), label)


def read_json(path):
    packed = path.read_bytes()
    return json.loads(gzip.decompress(packed) if path.suffix == '.gz' else packed)


def close(a, b, label):
    require(math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), label)


def rank_order(items, label):
    require(isinstance(items, list) and [x['rank'] for x in items] == list(range(8)), label)


def timing(record, label):
    raw = record['raw']
    require(len(raw) == 10, label + ': expected 10 samples')
    for sample, row in enumerate(raw):
        require(row['sample'] == sample and row['replays'] == 5, label + ': sample/replay count')
        rank_order(row['per_rank'], label + ': rank coverage')
        for kind in ('cuda_ms', 'host_ms'):
            values = [x[kind] for x in row['per_rank']]
            require(all(math.isfinite(x) and x > 0 for x in values), label + ': invalid timing')
            close(max(values), row['max_rank_' + kind], label + ': raw maximum')
    rank_order(record['per_rank_medians'], label + ': median rank coverage')
    result = {}
    for kind in ('cuda_ms', 'host_ms'):
        medians = [statistics.median(row['per_rank'][rank][kind] for row in raw)
                   for rank in range(8)]
        for rank, median in enumerate(medians):
            close(median, record['per_rank_medians'][rank][kind], label + ': rank median')
        maximum = max(medians)
        close(maximum, record['max_rank_median_' + kind], label + ': maximum rank median')
        result[kind.replace('_ms', '_us')] = maximum * 1000
        result['rank_medians_' + kind.replace('_ms', '_us')] = [x * 1000 for x in medians]
    close(statistics.median(row['max_rank_cuda_ms'] for row in raw),
          record['median_sample_max_rank_cuda_ms'], label + ': alternate timing statistic')
    return result


def matrix(value, label):
    require(len(value) == 8 and all(len(row) == 8 for row in value), label + ': matrix shape')
    require(all(type(x) is int and x >= 0 for row in value for x in row), label + ': matrix counts')
    require(all(value[r][r] == 0 for r in range(8)), label + ': diagonal')
    return sum(map(sum, value))


def audit(record, optimized, label):
    require(record['passed'] is True and record['checks'] and
            all(x is True for x in record['checks'].values()), label + ': failed audit')
    require(record['loads'] == [SHAPE['S'] * SHAPE['K']] * 8, label + ': unbalanced load')
    require(matrix(record['dispatch_matrix'], label) == record['remote_token_destinations'],
            label + ': remote count')
    require(matrix(record['home_to_replica_matrix'], label) == record['expert_copies'],
            label + ': expert count')
    copies = record['expert_copies_by_rank']
    require(len(copies) == 8 and all(type(x) is int and 0 <= x <= 8 for x in copies),
            label + ': replica budget')
    require(copies == [sum(row[r] for row in record['home_to_replica_matrix']) for r in range(8)]
            and sum(copies) == record['expert_copies'], label + ': replica matrix totals')
    dispatch = record['dispatch_matrix']
    endpoint = max(max(map(sum, dispatch)), max(sum(row[r] for row in dispatch) for r in range(8)))
    require(endpoint == record['max_endpoint_remote_tokens'], label + ': endpoint maximum')
    digest(record['owner_sha256'], label + ': owner hash')
    if optimized:
        require(record['status_by_rank'] == [0] * 8 and record['fell_back_to_native'] is False
                and record['optimizer_gain_claim_allowed'] is True, label + ': fallback/invalid status')


def trace(record, evidence, cohort, limit):
    rank = record['rank']
    label = f'{cohort}: trace rank{rank}'
    directory = cohort.replace('_03', '_traces_03')
    expected = Path(directory) / f'rank{rank}_graph.json.gz'
    require(Path(*Path(record['path']).parts[-2:]) == expected, label + ': trace filename')
    packed = (evidence / expected).read_bytes()
    require(sha(packed) == record['sha256'], label + ': compressed hash')
    raw = gzip.decompress(packed)
    require(sha(raw) == record['raw_sha256'], label + ': uncompressed hash')
    require(record['replays'] == 3 and record['sm_limit'] == limit and record['passed'] is True,
            label + ': trace metadata')
    events = json.loads(raw)['traceEvents']
    kernels = [x for x in events if x.get('cat') == 'kernel' and x.get('ph') == 'X']
    require(kernels, label + ': empty kernel trace')
    launches, timeline = [], []
    for kernel in kernels:
        args = kernel['args']
        grid = args['grid']
        require(isinstance(grid, list) and len(grid) == 3 and
                all(type(x) is int and x > 0 for x in grid), label + ': invalid grid')
        blocks = math.prod(grid)
        require(math.isfinite(kernel['ts']) and math.isfinite(kernel['dur']) and kernel['dur'] > 0,
                label + ': invalid kernel interval')
        launches.append(dict(name=kernel['name'], grid=grid, blocks=blocks, block=args.get('block'),
                             stream=args.get('stream'), duration_us=kernel['dur']))
        timeline.extend(((kernel['ts'], 1, blocks), (kernel['ts'] + kernel['dur'], 0, -blocks)))
    require(launches == record['launches'], label + ': raw launches disagree with summary')
    live = peak = 0
    for _, _, delta in sorted(timeline):
        live += delta
        require(live >= 0, label + ': invalid concurrent timeline')
        peak = max(peak, live)
    require(live == 0, label + ': unclosed timeline')
    names = [x['name'] for x in launches]
    anchors = dict(
        sample_gather=sum('gather_local(' in n for n in names),
        expert_greedy=sum('global_greedy_table_active<' in n for n in names),
        exchange_proposal=sum('propose_kernel(' in n or 'propose_ordinals_fused(' in n for n in names),
        exchange_prefix=sum('scan_blocks(' in n or 'scan_blocks_warp(' in n for n in names),
        layout_encode=sum('encode_kernel(' in n or 'bounded_materialize::encode(' in n for n in names),
        publication=sum('publish_kernel(' in n or 'bounded_materialize::publish(' in n for n in names))
    maximum = max(x['blocks'] for x in launches)
    nccl = sum('nccl' in n.lower() for n in names)
    observed = dict(kernel_count=len(launches), max_launch_blocks=maximum,
                    max_simultaneous_launched_blocks=peak,
                    streams=sorted({x['stream'] for x in launches}), nccl_kernel_count=nccl, anchors=anchors)
    for key, value in observed.items():
        require(record[key] == value, label + ': ' + key)
    require(anchors == ANCHORS, label + ': incomplete three-replay capture')
    if limit:
        require(maximum <= limit and peak <= limit and nccl == 0, label + ': exceeded SM budget/NCCL present')
    else:
        require(nccl > 0, label + ': baseline NCCL path absent')
    return dict(rank=rank, path=expected.as_posix(), **observed)


def test_logs(evidence):
    reports = {}
    # Each distributed process prints its own pytest summary; do not count them as distinct tests.
    for name, count, processes in [('config', 24, 1), ('components', 9, 1),
                                    ('public', 15, 8), ('compatibility', 5, 8)]:
        require((evidence / f'{name}_03.exit').read_text().strip() == '0', name + ': nonzero/missing exit')
        log = (evidence / f'{name}_03.log').read_text()
        summaries = re.findall(r'(?m)^\s*(\d+) passed[^\n]*\bin [\d.]+s', log)
        require(len(summaries) == processes and all(int(x) == count for x in summaries),
                name + ': missing or unexpected pytest result')
        require(not re.search(r'(?m)^.*\b\d+ (failed|error|errors|skipped)\b', log), name + ': test failure/skip')
        reports[name] = dict(passed_per_process=count, process_summaries=processes, exit_code=0)
    for sentinel in ('correctness_complete_03', 'matrix_complete_03'):
        require((evidence / sentinel).is_file(), 'missing ' + sentinel)
    return reports


def validate_cohort(name, data, evidence, oracle, oracle_hash):
    limit, fuse = FORMAL[name]
    require(data['status'] == 'complete' and data['failures'] == [], name + ': incomplete or failed run')
    require((evidence / (name + '.exit')).read_text().strip() == '0', name + ': nonzero exit')
    require((evidence / (name + '.log')).is_file(), name + ': missing raw log')
    meta = data['metadata']
    require(all(meta[k] == v for k, v in SHAPE.items()), name + ': unexpected workload')
    require(meta['samples'] == 10 and meta['replays'] == 5, name + ': timing protocol')
    require(meta['sm_limit'] == limit and meta['fuse'] is fuse, name + ': wrong variant')
    require(all(meta['planner_config'][k] == v for k, v in CONFIG.items()), name + ': planner config')
    require(meta['planner_config'].get('sm_limit', 0) == limit and
            meta['planner_config'].get('fuse', False) is fuse, name + ': config/metadata disagree')
    require(meta['oracle_sha256'] == oracle_hash, name + ': wrong CPU oracle')
    require(meta['device_sms'] >= limit, name + ': device smaller than cap')
    rank_order(meta['identities'], name + ': identity ranks')
    require(len({x['host'] for x in meta['identities']}) == 1 and
            len({x['uuid'] for x in meta['identities']}) == 8 and
            sorted(x['local_device'] for x in meta['identities']) == list(range(8)),
            name + ': expected eight distinct GPUs on one host')
    require(meta['source_files'], name + ': no source provenance')
    for source, value in meta['source_files'].items():
        require(not Path(source).is_absolute() and '..' not in Path(source).parts, name + ': invalid source path')
        digest(value, name + ': invalid source hash')
    layers = data['layers']
    require([x['layer'] for x in layers] == list(range(8)), name + ': incomplete layers')
    result = []
    for row in layers:
        layer = row['layer']; label = f'{name}: layer{layer}'
        digest(row['fixture_sha256'], label + ': fixture hash')
        for flag in ('graph_outputs_exact', 'all_rank_history_exact', 'cpu_oracle_checked'):
            require(row[flag] is True, label + ': ' + flag)
        audit(row['native_audit'], False, label + ': native')
        audit(row['audit'], True, label + ': sampled')
        require(row['audit']['remote_token_destinations'] <= row['native_audit']['remote_token_destinations'],
                label + ': communication regression')
        hashes = row['output_hashes_by_rank']
        require(len(hashes) == 8, label + ': missing output ranks')
        for h in hashes:
            require(set(h) == FIELDS, label + ': incomplete output fields')
            for key, value in h.items():
                digest(value, label + ': invalid ' + key + ' hash')
        for key in ('support', 'history', 'experts_to_copy'):
            require(all(h[key] == hashes[0][key] for h in hashes), label + ': global ' + key)
        expected = oracle[f'layer_{layer:02d}_cap128']
        diag = row['candidate_diagnostics']
        require(all(diag[k] == 128 for k in ('candidate_limit', 'effective_candidate_limit', 'retained_P')),
                label + ': candidate limit')
        ids = sorted(diag['selected_ids'])
        require(len(ids) == 128 and len(set(ids)) == 128 and
                all(type(x) is int and 0 <= x < diag['full_P'] for x in ids), label + ': retained IDs')
        require(expected['retained_ids']['shape'] == [128] and
                sha(struct.pack('<128i', *ids)) == expected['retained_ids']['sha256'], label + ': CPU retained IDs')
        require(expected['support']['shape'] == [8, 352] and
                hashes[0]['support'] == expected['support']['sha256'], label + ': CPU support')
        require(expected['history']['shape'][0] == diag['history_I'] and
                hashes[0]['history'] == expected['history']['sha256'], label + ': CPU history')
        require(expected['owner_rounds4']['shape'] == [8, 16768, 8] and
                row['audit']['owner_sha256'] == expected['owner_rounds4']['sha256'], label + ': CPU owners')
        require(set(row['variants']) == {'native', 'sampled'}, label + ': timing variants')
        timings = {key: timing(value, label + ': ' + key) for key, value in row['variants'].items()}
        traces = []
        if layer == 0 and name != 'baseline_after_03':
            rank_order(row['budget_traces'], label + ': trace ranks')
            traces = [trace(x, evidence, name, limit) for x in row['budget_traces']]
        else:
            require(not row.get('budget_traces'), label + ': unexpected trace cohort')
        native = row['native_audit']['remote_token_destinations']
        remote = row['audit']['remote_token_destinations']
        result.append(dict(layer=layer, timings=timings, traces=traces,
                           native_remote_token_destinations=native, remote_token_destinations=remote,
                           remote_reduction_percent=100 * (1 - remote / native),
                           native_expert_copies=row['native_audit']['expert_copies'],
                           expert_copies=row['audit']['expert_copies'], history_iterations=diag['history_I'],
                           full_candidates=diag['full_P'], retained_candidates=diag['retained_P']))
    return dict(sm_limit=limit, fuse=fuse, layers=result,
                mean_cuda_us=statistics.mean(x['timings']['sampled']['cuda_us'] for x in result),
                mean_host_us=statistics.mean(x['timings']['sampled']['host_us'] for x in result),
                native_mean_cuda_us=statistics.mean(x['timings']['native']['cuda_us'] for x in result))


def inventory(evidence):
    result = {}
    for path in sorted(evidence.rglob('*')):
        require(not path.is_symlink(), 'evidence symlink: ' + str(path))
        if path.is_file():
            require(path.suffix != '.npz', 'private router fixture must not be published: ' + str(path))
            result[path.relative_to(evidence).as_posix()] = sha(path.read_bytes())
    return result


def summarize(evidence, oracle_path, check_sources):
    oracle = read_json(oracle_path); oracle_hash = sha(oracle_path.read_bytes())
    tests = test_logs(evidence)
    records, results, inputs = {}, {}, {}
    for name in FORMAL:
        paths = [p for p in (evidence / (name + '.json'), evidence / (name + '.json.gz')) if p.is_file()]
        require(len(paths) == 1, name + ': need exactly one .json or .json.gz')
        data = read_json(paths[0]); records[name] = data
        inputs[name] = dict(path=paths[0].name, sha256=sha(paths[0].read_bytes()))
        results[name] = validate_cohort(name, data, evidence, oracle, oracle_hash)
    baseline = records['baseline_before_03']
    for name, data in records.items():
        for key in ('identities', 'gpu', 'torch', 'cuda', 'nccl', 'device_sms', 'included', 'excluded'):
            require(data['metadata'][key] == baseline['metadata'][key], name + ': environment/boundary changed: ' + key)
        for row, old in zip(data['layers'], baseline['layers']):
            for key in ('fixture_sha256', 'output_hashes_by_rank', 'native_audit', 'audit', 'candidate_diagnostics'):
                require(row[key] == old[key], f'{name}: layer{row["layer"]} differs from baseline: {key}')
        source_base = 'baseline_before_03' if not FORMAL[name][0] else 'bounded32_03'
        require(data['metadata']['source_files'] == records[source_base]['metadata']['source_files'],
                name + ': source changed within cohort')
    if check_sources:
        for source, value in records['bounded32_03']['metadata']['source_files'].items():
            require(sha((ROOT / source).read_bytes()) == value, 'current source differs from measured source: ' + source)
    before = results['baseline_before_03']; after = results['baseline_after_03']
    reference = (before['mean_cuda_us'] + after['mean_cuda_us']) / 2
    reference_layers = [(a['timings']['sampled']['cuda_us'] + b['timings']['sampled']['cuda_us']) / 2
                        for a, b in zip(before['layers'], after['layers'])]
    for name, result in results.items():
        result.update(latency_change_vs_bracketing_baseline_percent=100 * (result['mean_cuda_us'] / reference - 1),
                      latency_change_vs_before_percent=100 * (result['mean_cuda_us'] / before['mean_cuda_us'] - 1),
                      latency_change_vs_after_percent=100 * (result['mean_cuda_us'] / after['mean_cuda_us'] - 1))
        for row, ref in zip(result['layers'], reference_layers):
            row['latency_change_vs_bracketing_baseline_percent'] = 100 * (row['timings']['sampled']['cuda_us'] / ref - 1)
    native = sum(x['native_remote_token_destinations'] for x in before['layers'])
    remote = sum(x['remote_token_destinations'] for x in before['layers'])
    copies = sum(x['expert_copies'] for x in before['layers'])
    native_copies = sum(x['native_expert_copies'] for x in before['layers'])
    files = inventory(evidence)
    return dict(format_version=1, validation='passed', formal_inputs=inputs,
                hardware={k: baseline['metadata'][k] for k in ('gpu', 'device_sms', 'identities', 'torch', 'cuda', 'nccl')},
                workload=SHAPE, planner_config=CONFIG, tests=tests,
                fixture_sha256={f'layer_{r["layer"]:02d}.npz': r['fixture_sha256'] for r in baseline['layers']},
                oracle_sha256=oracle_hash, all_outputs_exact=True, all_audits_passed=True, fallback_count=0,
                current_sources_checked=check_sources,
                timing_method='For each layer: median of 10 CUDA-event samples per rank; max across 8 ranks; each sample averages 5 Graph replays. Then arithmetic mean over 8 layers.',
                baseline_reference=dict(mean_cuda_us=reference, layer_cuda_us=reference_layers,
                    definition='Arithmetic mean of before and after baseline timings; not a randomized or drift-corrected estimate.',
                    before_to_after_change_percent=100 * (after['mean_cuda_us'] / before['mean_cuda_us'] - 1),
                    layer_before_to_after_change_percent=[100 * (b['timings']['sampled']['cuda_us'] / a['timings']['sampled']['cuda_us'] - 1)
                                                         for a, b in zip(before['layers'], after['layers'])]),
                communication=dict(native_remote_token_destinations=native, remote_token_destinations=remote,
                    remote_reduction_percent=100 * (1 - remote / native), native_expert_copies=native_copies,
                    expert_copies=copies, additional_copies_vs_native=copies-native_copies,
                    additional_copies_vs_previous_sampled=0, all_variants_exact=True,
                    interpretation='Counts sum 8 independent layers. A remote token-destination counts once per destination GPU; it is not a compute pair or a measured payload transfer time.'),
                variants=results,
                exploratory_files=[name for name in files if name.startswith(('probe', 'launcher')) or '_01' in name or '_02' in name],
                limitations=[
                    'Planner-only latency; excludes expert weights, token activations, gradients, GEMM, optimizer, histogram building, allocation, JIT and audits.',
                    'Communication reduction is exact route-plan accounting. This cohort does not measure end-to-end training speed or weight/gradient payload costs.',
                    'SM upper bound uses conservative concurrent launched CTA counts from raw GPU trace intervals. It does not pin SM IDs or measure achieved occupancy.',
                    'Budget traces cover layer 0 on all 8 ranks and 3 Graph replays; full-output and CPU-oracle checks cover all 8 layers.',
                    'Runs are sequential with bracketing baselines, not randomized repeated cohorts or confidence intervals.',
                    'Exploratory probes are retained verbatim and excluded from every formal timing table.',
                    'Private router NPZ files are omitted; their original SHA-256 and independent CPU array hashes are retained.']), files


def markdown(summary):
    ref = summary['baseline_reference']; comm = summary['communication']
    lines = ['# 32-SM planner: frozen 03 results', '',
             'All formal records, original GPU traces, output hashes, CPU-oracle hashes, and test exit records passed independent validation.', '',
             'Hardware: one omni-dev host, 8 × NVIDIA H200 (132 SM/GPU). Workload: 134,144 tokens total, E=352, K=8; sample size 256, candidate cap 128, four exchange rounds.', '',
             'Latency in microseconds. Each layer uses the maximum of eight rank medians (10 samples, 5 Graph replays/sample); the headline is the mean of eight layers.', '',
             '| Variant | Mean GPU µs | Mean host µs | Change vs bracketing baseline | Max launch CTAs | Peak concurrent launched CTAs | Kernels/replay/rank |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, result in summary['variants'].items():
        traces = result['layers'][0]['traces']
        launches = str(max(x['max_launch_blocks'] for x in traces)) if traces else '—'
        peak = str(max(x['max_simultaneous_launched_blocks'] for x in traces)) if traces else '—'
        kernels = sorted({x['kernel_count'] // 3 for x in traces})
        lines.append(f'| {name} | {result["mean_cuda_us"]:.2f} | {result["mean_host_us"]:.2f} | '
                     f'{result["latency_change_vs_bracketing_baseline_percent"]:+.2f}% | {launches} | {peak} | '
                     + ('/'.join(map(str, kernels)) if kernels else '—') + ' |')
    lines += ['', f'Bracketing baseline: {ref["mean_cuda_us"]:.2f} µs. Before→after drift: '
              f'{ref["before_to_after_change_percent"]:+.3f}%. Positive change means slower. '
              'This arithmetic reference does not remove drift or provide a confidence interval.', '',
              '| Layer | Before | Bounded 32 | Fused 32 | Fused 48 | Fused 64 | Fused 96 | Fused 128 | After |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for layer in range(8):
        values = [result['layers'][layer]['timings']['sampled']['cuda_us'] for result in summary['variants'].values()]
        lines.append('| ' + str(layer) + ' | ' + ' | '.join(f'{x:.2f}' for x in values) + ' |')
    lines += ['', f'All variants reproduce the previous sampled planner exactly: all nine output arrays on every rank, expert support, selected candidate IDs, greedy history, owners, addresses, and source metadata. No formal run fell back.', '',
              f'Across the eight layers, remote token-destinations: {comm["native_remote_token_destinations"]:,} native MoonEP → '
              f'{comm["remote_token_destinations"]:,} sampled planner, a {comm["remote_reduction_percent"]:.5f}% reduction. '
              f'Expert copies: {comm["native_expert_copies"]} native → {comm["expert_copies"]} sampled '
              f'({comm["additional_copies_vs_native"]:+d}); the SM/fusion changes add zero copies versus the previous sampled planner.', '',
              '| Layer | Native remote destinations | Sampled remote destinations | Reduction | Native copies | Sampled copies | Greedy selections |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['variants']['baseline_before_03']['layers']:
        lines.append(f'| {row["layer"]} | {row["native_remote_token_destinations"]:,} | '
                     f'{row["remote_token_destinations"]:,} | {row["remote_reduction_percent"]:.2f}% | '
                     f'{row["native_expert_copies"]} | {row["expert_copies"]} | {row["history_iterations"]} |')
    lines += ['', 'Correctness suites:', '']
    for name, result in summary['tests'].items():
        lines.append(f'- {name}: {result["passed_per_process"]} passed per process; '
                     f'{result["process_summaries"]} process summaries; exit 0.')
    lines += ['', 'Evidence limits:', ''] + ['- ' + x for x in summary['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=HERE / 'evidence')
    parser.add_argument('--oracle', type=Path, default=HERE.parent / 'candidate_limit128' / 'cpu_array_hashes.json')
    parser.add_argument('--manifest', type=Path, default=HERE / 'evidence_manifest.json')
    parser.add_argument('--write-manifest', action='store_true', help='explicitly seal current evidence after all semantic checks pass')
    parser.add_argument('--check-sources', action='store_true', help='require current code hashes to equal the measured bounded cohort')
    parser.add_argument('--output', type=Path, help='write summary JSON outside evidence; otherwise print JSON')
    parser.add_argument('--markdown', type=Path, help='also write human-readable summary outside evidence')
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    for path in (args.manifest, args.output, args.markdown):
        if path:
            require(not path.resolve().is_relative_to(evidence), 'generated outputs must be outside evidence')
    summary, files = summarize(evidence, args.oracle, args.check_sources)
    manifest = {'evidence/' + name: value for name, value in files.items()}
    for attempt in ('01', '02', '03'):
        path = Path('implementation_attempts') / f'payload_{attempt}.tar.gz'
        manifest[path.as_posix()] = sha((HERE / path).read_bytes())
    if args.write_manifest:
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    else:
        require(read_json(args.manifest) == manifest, 'evidence inventory or hashes differ from sealed manifest')
    summary['evidence_manifest_sha256'] = sha(args.manifest.read_bytes())
    summary['evidence_file_count'] = len(files)
    text = json.dumps(summary, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    if args.markdown:
        args.markdown.write_text(markdown(summary))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, IndexError, struct.error) as exc:
        print('VALIDATION FAILED: ' + str(exc), file=sys.stderr)
        sys.exit(1)
