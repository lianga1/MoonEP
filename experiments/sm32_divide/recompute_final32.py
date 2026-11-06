"""Independent final SM32 audit using only Python's standard library.

Run after final03/evidence has been downloaded:
  python3 recompute_final32.py

No analysis result is written while required evidence is absent or invalid.
Raw profiler kernel durations remain diagnostic; formal latency comes only
from untouched production Graph measurements. Baseline evidence is frozen.
"""
from collections import defaultdict
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT/'baseline_profiles_02/analysis_sources'
PLAN_FIELDS = ('dst', 'experts_to_copy', 'zero_fill_ranges', 'remote_stats',
               'cu', 'owner', 'support', 'source_info', 'history')
BENCHMARKS = ('control_bounded_before03', 'control_fused_before03',
              'control_bounded_after03', 'control_fused_after03',
              'final_bounded03', 'final_selective03')
PROFILES = ('profile_bounded03', 'profile_selective03')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as file:
        return json.load(file)


def close(a, b, tolerance=1e-5):
    assert abs(a-b) <= tolerance, (a, b, a-b)


def equal_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a: equal_tree(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    elif isinstance(a, float): close(a, b, 1e-8)
    else: assert a == b, (a, b)


def mean_dict(values):
    keys = set().union(*(set(x) for x in values))
    return {key: statistics.mean(x.get(key, 0.) for x in values) for key in sorted(keys)}


def timing(record):
    raw = record['raw']
    assert raw and len(record['per_rank_medians']) == 8
    medians = []
    for rank in range(8):
        for sample in raw:
            assert len(sample['per_rank']) == 8 and sample['per_rank'][rank]['rank'] == rank
            assert sample['replays'] > 0 and sample['per_rank'][rank]['cuda_ms'] > 0
        value = statistics.median(x['per_rank'][rank]['cuda_ms'] for x in raw)*1000
        close(value, record['per_rank_medians'][rank]['cuda_ms']*1000)
        medians.append(value)
    close(max(medians), record['max_rank_median_cuda_ms']*1000)
    return dict(max_rank_median_us=max(medians), rank_median_us=medians,
                samples=len(raw), replays=raw[0]['replays'], raw=raw)


def source_check(metadata, expected):
    observed = metadata['source_files']
    assert observed and any(name.startswith('moonep/') for name in observed)
    for name, value in observed.items():
        assert name in expected and value == expected[name], ('source mismatch', name, value, expected.get(name))
    return len(observed)


def config_check(metadata):
    cfg = metadata.get('planner_config', metadata.get('config'))
    assert cfg is not None
    assert all(cfg[key] == value for key, value in dict(candidate_limit=128, rounds=4,
        sample_size=256, groups_per_class=2, sm_limit=32).items()), cfg
    for key, value in dict(R=8, N=134144, S=16768, E=352, K=8, B=8).items():
        assert metadata[key] == value, (key, metadata[key])
    identities = metadata.get('identities', metadata.get('devices'))
    assert len(identities) == 8 and len({x['uuid'] for x in identities}) == 8
    assert len({x['host'] for x in identities}) == 1
    return cfg


def trace_path(evidence, record):
    original = Path(record['path'])
    path = evidence/original.parent.name/original.name
    assert path.is_file(), path
    assert sha(path) == record['sha256'], ('compressed trace SHA', path)
    raw = gzip.decompress(path.read_bytes())
    if 'raw_sha256' in record:
        assert hashlib.sha256(raw).hexdigest() == record['raw_sha256'], ('raw trace SHA', path)
    if 'raw_json_bytes' in record: assert len(raw) == record['raw_json_bytes']
    return json.loads(raw), dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                               raw_sha256=hashlib.sha256(raw).hexdigest())


def budget_audit(data):
    """Independent concurrent launched-CTA upper bound; never SM-ID occupancy."""
    kernels = [x for x in data['traceEvents'] if x.get('cat') == 'kernel' and x.get('ph') == 'X']
    assert kernels
    endpoints, blocks, streams, names = [], [], set(), []
    for item in kernels:
        args = item['args']; grid = args['grid']
        assert len(grid) == 3 and all(isinstance(x, int) and x > 0 for x in grid)
        count = math.prod(grid)
        assert count <= 32, ('grid limit', count, item['name'])
        assert 'nccl' not in item['name'].lower(), item['name']
        endpoints.extend(((item['ts'], 1, count), (item['ts']+item['dur'], 0, -count)))
        blocks.append(count); streams.add(args['stream']); names.append(item['name'])
    current = peak = 0
    for _, _, delta in sorted(endpoints):
        current += delta; peak = max(peak, current)
        assert current >= 0
    assert current == 0 and peak <= 32, ('concurrent CTA limit', peak)
    anchors = dict(
        sample_gather=sum('gather_local(' in n for n in names),
        expert_greedy=sum('global_greedy_table_active<' in n or 'global_greedy_compact_active<' in n for n in names),
        exchange_proposal=sum('propose_kernel(' in n or 'propose_ordinals_fused(' in n or 'propose_ordinals_fused<' in n for n in names),
        exchange_prefix=sum('scan_blocks(' in n or 'scan_blocks_warp(' in n for n in names),
        layout_encode=sum('encode_kernel(' in n or 'bounded_materialize::encode(' in n for n in names),
        publication=sum('publish_kernel(' in n or 'bounded_materialize::publish(' in n for n in names))
    assert anchors == dict(sample_gather=3, expert_greedy=3, exchange_proposal=12,
                           exchange_prefix=12, layout_encode=3, publication=3), anchors
    return dict(passed=True, kernel_count=len(kernels), max_launch_blocks=max(blocks),
        max_simultaneous_launched_blocks=peak, nccl_kernel_count=0, streams=sorted(streams), anchors=anchors)


def stage_group(stage):
    if stage == 'residency_frequency_collective': return 'frequency_collective'
    if stage == 'residency_frequency_shard': return 'frequency_compute'
    if stage == 'residency_frequency_unpack': return 'frequency_unpack'
    if stage == 'native_seed': return 'native_seed'
    if stage == 'pre_sync': return 'entry_sync'
    if stage in ('input_copy', 'decode_owner'): return 'input_copy_decode'
    if stage.startswith('residency_'):
        return 'sample_collective' if 'collective' in stage else 'residency_local'
    if stage.startswith('exchange_'): return 'exchange_setup'
    if stage.startswith('round'): return 'exchange_'+stage.split('_', 1)[1]
    if stage.startswith('materialize_'):
        return 'materialize_collective' if 'collective' in stage else 'materialize_local'
    if stage.startswith('finish_'):
        return 'finish_collective' if 'collective' in stage else 'finish_local'
    if stage.startswith('publication_'): return 'publication'
    raise AssertionError(stage)


def kernel_key(stage, name):
    if stage == 'native_seed': return 'native_seed/mixed'
    if stage == 'pre_sync' or stage.startswith('publication_barrier'):
        return 'native_barriers/signal_and_wait'
    simple = name.replace('(anonymous namespace)::', '').replace('void ', '').split('(')[0]
    return stage_group(stage)+'/'+simple


def representative(ranks):
    """Select one rank, then the complete observation(s) forming its median."""
    rank = max(range(8), key=lambda r: statistics.median(x['total_us'] for x in ranks[r]))
    rows = sorted(ranks[rank], key=lambda x: x['total_us'])
    middle = [rows[len(rows)//2]] if len(rows)%2 else rows[len(rows)//2-1:len(rows)//2+1]
    result = dict(rank=rank, observations=[x['replay'] for x in middle],
                  total_us=statistics.mean(x['total_us'] for x in middle))
    for name in ('stage_us', 'group_us', 'category_us', 'kernel_us'):
        if name in middle[0]: result[name] = mean_dict([x[name] for x in middle])
    for name in ('gap_us', 'kernel_sum_us'):
        if name in middle[0]: result[name] = statistics.mean(x[name] for x in middle)
    return result


def check_audit(audit):
    assert audit['passed'] and all(audit['checks'].values())
    if 'status_by_rank' in audit:
        assert audit['status_by_rank'] == [0]*8 and not audit['fell_back_to_native']
    assert audit['loads'] == [134144]*8


def core_hash_check(items, original, cpu, audit, counter):
    assert len(items) == 8
    for rank, record in enumerate(items):
        for field in PLAN_FIELDS:
            value = record[field]
            digest = value['sha256'] if isinstance(value, dict) else value
            assert digest == original['output_hashes_by_rank'][rank][field], (rank, field, 'original 802 plan mismatch')
            counter['plan9_hash_comparisons'] += 1
        for field in ('support', 'history'):
            value = record[field]
            digest = value['sha256'] if isinstance(value, dict) else value
            assert digest == cpu[field]['sha256'], (rank, field, 'CPU oracle mismatch')
            if isinstance(value, dict): assert value == cpu[field]
            counter['cpu_hash_comparisons'] += 1
    assert audit['owner_sha256'] == cpu['owner_rounds4']['sha256']
    counter['cpu_hash_comparisons'] += 1


def benchmark(evidence, name, source_hashes, originals, oracle, counters):
    path = evidence/(name+'.json'); data = load(path)
    assert data['status'] == 'complete' and not data['failures'], name
    assert [x['layer'] for x in data['layers']] == list(range(8)), name
    assert (evidence/(name+'.exit')).read_text().strip() == '0'
    cfg = config_check(data['metadata']); counters['source_hash_comparisons'] += source_check(data['metadata'], source_hashes)
    rows, traces = [], []
    for row in data['layers']:
        layer = row['layer']; original = originals[layer]
        assert row['fixture_sha256'] == original['fixture_sha256']
        assert row['graph_outputs_exact'] and row['all_rank_history_exact'] and row['cpu_oracle_checked']
        check_audit(row['native_audit']); check_audit(row['audit'])
        cpu = oracle[f'layer_{layer:02d}_cap128']
        core_hash_check(row['output_hashes_by_rank'], original, cpu, row['audit'], counters)
        ids = sorted(row['candidate_diagnostics']['selected_ids'])
        assert len(ids) == 128 and len(set(ids)) == 128
        encoded = struct.pack('<'+'i'*len(ids), *ids)
        assert hashlib.sha256(encoded).hexdigest() == cpu['retained_ids']['sha256']
        counters['cpu_hash_comparisons'] += 1
        assert row['audit']['remote_token_destinations'] == original['audit']['remote_token_destinations']
        assert row['audit']['expert_copies'] == original['audit']['expert_copies']
        rows.append(dict(layer=layer, timing=timing(row['variants']['sampled']),
            native_timing=timing(row['variants']['native']),
            remote=row['audit']['remote_token_destinations'], native_remote=row['native_audit']['remote_token_destinations'],
            copies=row['audit']['expert_copies'], output_hashes_by_rank=row['output_hashes_by_rank']))
        for saved in row.get('budget_traces', []):
            raw, provenance = trace_path(evidence, saved); rebuilt = budget_audit(raw)
            for key, value in rebuilt.items(): assert saved[key] == value, (name, key)
            counters['budget_traces'] += 1
            traces.append(dict(layer=layer, rank=saved['rank'], **rebuilt, **provenance))
    if name.startswith('final_'):
        assert len(traces) == 8 and {x['rank'] for x in traces} == set(range(8)), ('missing budget traces', name)
    return dict(name=name, sha256=sha(path), config=cfg, metadata=data['metadata'], layers=rows,
        mean_layer_us=statistics.mean(x['timing']['max_rank_median_us'] for x in rows),
        native_mean_layer_us=statistics.mean(x['native_timing']['max_rank_median_us'] for x in rows), budget_traces=traces)


def profile(evidence, name, source_hashes, originals, oracle, parser, counters):
    path = evidence/(name+'.json'); data = load(path)
    assert data['status'] == 'complete' and not data['failures']
    assert [x['layer'] for x in data['layers']] == list(range(8))
    assert (evidence/(name+'.exit')).read_text().strip() == '0'
    cfg = config_check(data['metadata']); counters['source_hash_comparisons'] += source_check(data['metadata'], source_hashes)
    assert data['metadata']['args']['trace_discard'] == 2 and data['metadata']['args']['trace_replays'] == 5
    assert data['metadata']['args']['alignment_replays'] >= 2
    assert data['metadata']['source_files']['tests/profile_sampled_sm32_trace.py'] == sha(FROZEN/'tests/profile_sampled_sm32_trace.py')
    rank_raw = [load(evidence/f'{name}_rank{r}_raw.json') for r in range(8)]
    rows, all_observations = [], []
    for row in data['layers']:
        layer = row['layer']; original = originals[layer]; check = row['correctness']
        assert row['completed'] and row['fixture_sha256'] == original['fixture_sha256']
        assert row['residency_frequency_sharding_enabled'] is True
        names = [x['name'] for x in row['stages']]
        assert len(names) == 40 and names[6:11] == ['residency_records_prepare', 'residency_frequency_shard',
            'residency_frequency_collective', 'residency_frequency_unpack', 'residency_candidate_finalize']
        assert row['stages'][8]['kind'] == 'peer_collective'
        check_audit(check['native']); check_audit(check['sampled'])
        for key in ('split_eager_exact', 'external_event_graph_exact', 'steady_profiler_graph_exact', 'formal_after_graph_exact'):
            assert len(check[key]) == 8 and all(not x['mismatches'] for x in check[key]), (name, layer, key)
        core_hash_check(check['production_output_hashes_by_rank'], original,
                        oracle[f'layer_{layer:02d}_cap128'], check['sampled'], counters)
        item = dict(layer=layer, formal_before=timing(row['formal_before']), formal_after=timing(row['formal_after']),
            source_snapshot_hashes=check['production_output_hashes_by_rank'], trace_by_rank=[], event_by_rank=[], trace_provenance=[])
        rows.append(item)
        for rank in range(8):
            assert rank_raw[rank]['layers'][layer]['stage_events'] == row['stage_events_by_rank'][rank]
            assert rank_raw[rank]['layers'][layer]['trace'] == row['traces_by_rank'][rank]
            event_rows = []
            for event in row['stage_events_by_rank'][rank]['raw']:
                total = sum(event['stage_us'].values()); error = event['total_us']-total
                close(total, event['sum_stage_us'], 1e-8); close(error, event['closure_error_us'], 1e-8)
                counters['event_max_closure_us'] = max(counters['event_max_closure_us'], abs(error))
                grouped = defaultdict(float)
                for stage, duration in event['stage_us'].items(): grouped[stage_group(stage)] += duration
                event_rows.append(dict(replay=event['sample'], total_us=event['total_us'], stage_us=event['stage_us'], group_us=dict(grouped)))
                counters['event_samples'] += 1
            item['event_by_rank'].append(event_rows)
            saved = row['traces_by_rank'][rank]
            eager, ep = trace_path(evidence, saved['eager_trace']); gpu_trace, gp = trace_path(evidence, saved['graph_trace'])
            counters['profile_trace_files'] += 2
            mapping = parser.eager_mapping(eager, row['stages']); assert mapping == saved['mapping']
            rebuilt = parser.summarize_graph(gpu_trace, mapping, 2, 5, 32)
            equal_tree(rebuilt['raw_replays'], saved['raw_replays'])
            item['trace_provenance'].append(dict(rank=rank, eager=ep, graph=gp, kernel_count_per_replay=rebuilt['kernel_count_per_replay']))
            retained = []
            for replay in rebuilt['raw_replays']:
                counters['profile_replays'] += 1
                assert replay['discarded'] == (replay['replay'] < 2)
                grouped, fine = defaultdict(float), defaultdict(float)
                for launch in replay['launches']:
                    grouped[stage_group(launch['stage'])] += launch['duration_us']
                    fine[kernel_key(launch['stage'], launch['name'])] += launch['duration_us']
                span, kernels, gap = replay['first_start_to_last_end_us'], sum(grouped.values()), replay['positive_inter_kernel_gaps_us']
                error = span-kernels-gap; close(error, 0, .01)
                counters['trace_max_closure_us'] = max(counters['trace_max_closure_us'], abs(error))
                if replay['discarded']: continue
                value = dict(replay=replay['replay'], total_us=span, kernel_sum_us=kernels, gap_us=gap,
                    stage_us=replay['stage_kernel_us'], group_us=dict(grouped), category_us=replay['category_kernel_us'], kernel_us=dict(fine))
                retained.append(value); all_observations.append(value); counters['retained_profile_replays'] += 1
            item['trace_by_rank'].append(retained)
        item['critical_trace'] = representative(item['trace_by_rank'])
        item['critical_events'] = representative(item['event_by_rank'])
        close(sum(item['critical_trace']['group_us'].values())+item['critical_trace']['gap_us'], item['critical_trace']['total_us'], .01)
    aggregate = dict(
        formal_before_us=statistics.mean(x['formal_before']['max_rank_median_us'] for x in rows),
        formal_after_us=statistics.mean(x['formal_after']['max_rank_median_us'] for x in rows),
        trace_span_us=statistics.mean(x['critical_trace']['total_us'] for x in rows),
        trace_kernel_us=statistics.mean(x['critical_trace']['kernel_sum_us'] for x in rows),
        trace_gap_us=statistics.mean(x['critical_trace']['gap_us'] for x in rows),
        event_total_us=statistics.mean(x['critical_events']['total_us'] for x in rows),
        trace_group_us=mean_dict([x['critical_trace']['group_us'] for x in rows]),
        trace_stage_us=mean_dict([x['critical_trace']['stage_us'] for x in rows]),
        trace_category_us=mean_dict([x['critical_trace']['category_us'] for x in rows]),
        trace_kernel_detail_us=mean_dict([x['critical_trace']['kernel_us'] for x in rows]),
        all_rank_replay_mean_span_us=statistics.mean(x['total_us'] for x in all_observations),
        all_rank_replay_mean_group_us=mean_dict([x['group_us'] for x in all_observations]))
    aggregate['formal_midpoint_us'] = (aggregate['formal_before_us']+aggregate['formal_after_us'])/2
    aggregate['event_minus_formal_midpoint_us'] = aggregate['event_total_us']-aggregate['formal_midpoint_us']
    aggregate['event_minus_trace_us'] = aggregate['event_total_us']-aggregate['trace_span_us']
    return dict(name=name, sha256=sha(path), config=cfg, metadata=data['metadata'], layers=rows, aggregate=aggregate)


def compare_baseline_phases(previous, profiles):
    """Cross-session stage evidence, kept separate from bracketing controls."""
    comparisons = {}
    def matching(values, fragment):
        return sum(value for key, value in values.items() if fragment in key)
    for old_variant, new_variant in [('bounded','bounded'),('fused','selective')]:
        old = previous['variants'][old_variant]['aggregate']
        new = profiles[f'profile_{new_variant}03']['aggregate']
        ok, nk = old['trace_kernel_detail_us'], new['trace_kernel_detail_us']
        old_frequency = matching(ok, '/record_frequencies_bounded')
        old_groups = dict(old['trace_group_us'])
        old_groups['residency_local'] -= old_frequency
        old_groups.update(frequency_compute=old_frequency, frequency_collective=0., frequency_unpack=0.)
        new_groups = new['trace_group_us']
        rows = {}
        def add(key, a, b): rows[key] = dict(before_us=a, after_us=b, delta_us=b-a)
        add('residency_complete_including_frequency_collective', old['trace_group_us']['residency_local'],
            sum(new_groups.get(key,0.) for key in ('residency_local','frequency_compute','frequency_collective','frequency_unpack')))
        add('frequency_complete', old_frequency,
            sum(new_groups.get(key,0.) for key in ('frequency_compute','frequency_collective','frequency_unpack')))
        add('greedy', matching(ok,'/global_greedy_'), matching(nk,'/global_greedy_'))
        add('materialize_local', old_groups['materialize_local'], new_groups['materialize_local'])
        if old_variant == 'bounded':
            add('materialize_input', matching(ok,'/bounded_materialize::input'), matching(nk,'/bounded_materialize::input'))
        else:
            add('materialize_input', matching(ok,'/bounded_materialize::prepare_fused'), matching(nk,'/bounded_materialize::prepare_fused'))
        add('histogram_prefix', matching(ok,'/bounded_materialize::histogram_prefix'), matching(nk,'/bounded_materialize::rank_prefix_warp<false>'))
        add('expert_starts', matching(ok,'/bounded_materialize::starts'), matching(nk,'/bounded_materialize::rank_prefix_warp<true>'))
        add('exchange_propose_ordinal_prefix', old_groups['exchange_propose_ordinal_prefix'],new_groups['exchange_propose_ordinal_prefix'])
        add('exchange_count_collective',old_groups['exchange_counts_collective'],new_groups['exchange_counts_collective'])
        add('trace_gap',old['trace_gap_us'],new['trace_gap_us'])
        add('trace_span',old['trace_span_us'],new['trace_span_us'])
        close(sum(old_groups.values())+old['trace_gap_us'],old['trace_span_us'],.01)
        close(sum(new_groups.values())+new['trace_gap_us'],new['trace_span_us'],.01)
        comparisons[new_variant] = dict(
            before_profile_formal_midpoint_us=old['formal_midpoint_us'],
            after_profile_formal_midpoint_us=new['formal_midpoint_us'],
            before_group_us=old_groups, after_group_us=new_groups,
            group_delta_us={key:new_groups.get(key,0.)-old_groups.get(key,0.) for key in sorted(set(old_groups)|set(new_groups))},
            detail=rows,
            warning='Earlier baseline02 profiling and final03 profiling are separate sessions. Their phase deltas are diagnostic evidence, not an exact causal decomposition of the same-session formal control gain. Detail rows overlap and must not be summed.')
    return comparisons


def markdown(result):
    formal, profiles, comparison = result['benchmarks'], result['profiles'], result['comparison']
    lines = ['# 最终 32 SM planner 独立复算', '',
        '本报告只使用已完成且通过核验的八卡正式结果。固定候选 128、四轮、R=8、E=352、K=8、全局 N=134144、采样 256；全部在同机 8 张 H200 上执行。', '',
        '## 无插桩完整 planner 延迟', '',
        '每层每 rank 先取原始样本中位数，再取最慢 rank，最后平均相同八层。单位 μs。控制组是冻结的 802 版；前后中点仅用于比较漂移，不是一次新测量。', '',
        '| 路径 | 控制前 | 控制后 | 控制中点 | 新版本正式 | 相对中点变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for old, new, label in [('bounded', 'bounded', '32 SM 未融合'), ('fused', 'selective', '32 SM 选择性融合')]:
        c = comparison[new]
        lines.append(f'| {label} | {c["control_before_us"]:.3f} | {c["control_after_us"]:.3f} | {c["control_midpoint_us"]:.3f} | {c["final_us"]:.3f} | {c["delta_us"]:+.3f} ({c["relative_change_percent"]:+.2f}%) |')
    lines += ['', '## 稳态 kernel 分解', '',
        '先在每层选择总 span 中位数最大的 rank，再取该 rank 的完整中位回放向量；所有阶段取自同一次回放，各行与空隙可以闭合。绝不累加各阶段独立的最大 rank。前两次 profiler 回放已丢弃，后五次用于统计。', '',
        '| 阶段 | 新未融合 | 新选择性融合 | 差额 |', '|---|---:|---:|---:|']
    a, b = profiles['profile_bounded03']['aggregate'], profiles['profile_selective03']['aggregate']
    labels = {'entry_sync':'入口同步', 'input_copy_decode':'输入复制与解码', 'native_seed':'原生 seed（计算/通信/等待混合）',
        'residency_local':'驻留记录准备与候选筛选/贪心', 'sample_collective':'样本表合并',
        'frequency_compute':'频率分片计算', 'frequency_collective':'新增频率表卡间收集', 'frequency_unpack':'频率解包',
        'exchange_setup':'交换初始化', 'exchange_propose_ordinal_prefix':'四轮提议/序号/前缀',
        'exchange_counts_collective':'四轮计数表通信', 'exchange_quota':'四轮配额', 'exchange_apply':'四轮应用',
        'materialize_local':'物理地址本地构建', 'materialize_collective':'物理地址计数表通信',
        'finish_local':'状态与最终方案选择', 'finish_collective':'最终状态通信', 'publication':'来源信息发布与同步'}
    for key in labels:
        av, bv = a['trace_group_us'].get(key,0.), b['trace_group_us'].get(key,0.)
        lines.append(f'| {labels[key]} | {av:.3f} | {bv:.3f} | {bv-av:+.3f} |')
    lines += [f'| kernel 间空隙 | {a["trace_gap_us"]:.3f} | {b["trace_gap_us"]:.3f} | {b["trace_gap_us"]-a["trace_gap_us"]:+.3f} |',
        f'| **完整 span** | **{a["trace_span_us"]:.3f}** | **{b["trace_span_us"]:.3f}** | **{b["trace_span_us"]-a["trace_span_us"]:+.3f}** |', '',
        '新增频率通信已单列，并包含在总时长中。原始细 kernel 表、全部 rank/replay 的均值、被选择的 rank 和回放编号均保存在 JSON。独立 barrier 时间包含信号和到达等待；融合通信与原生 seed 内部不能靠该 trace 再拆成纯计算、纯网络和纯等待。', '',
        '## 仪器开销单独列示', '', '| 测量 | 新未融合 | 新选择性融合 |', '|---|---:|---:|']
    for label, key in [('profile 无插桩前','formal_before_us'),('profile 无插桩后','formal_after_us'),
        ('阶段 event 总时长','event_total_us'),('event − 稳态 trace','event_minus_trace_us'),
        ('event − 无插桩前后中点','event_minus_formal_midpoint_us')]:
        lines.append(f'| {label} | {a[key]:.3f} | {b[key]:.3f} |')
    if 'baseline02_phase_comparison' in result:
        pb, ps = (result['baseline02_phase_comparison'][key]['detail'] for key in ('bounded','selective'))
        lines += ['', '## 相对优化前的阶段变化', '',
            '以下细项来自此前 02 稳态 trace 与本次 03 稳态 trace。两个批次的旧版正式基线存在漂移，因此不能将这些阶段差额相加，替代上面的同批次前后控制组收益。表中“驻留完整阶段”包含新增加的频率计算、通信和解包；其他细项包含在其所属大阶段内，不可重复相加。', '',
            '| 阶段/细项 | 旧未融合 | 新未融合 | 旧融合 | 新选择性融合 |', '|---|---:|---:|---:|---:|']
        names = [('residency_complete_including_frequency_collective','驻留完整阶段（含新增频率通信）'),
            ('frequency_complete','频率统计完整路径'),('greedy','贪心选择'),
            ('materialize_local','物理地址本地构建'),('materialize_input','input / prepare_fused'),
            ('histogram_prefix','rank 桶直方图前缀'),('expert_starts','专家段起点'),
            ('exchange_propose_ordinal_prefix','四轮提议/序号/前缀'),('exchange_count_collective','四轮计数表通信'),
            ('trace_gap','kernel 间空隙'),('trace_span','完整 trace span')]
        for key,label in names:
            lines.append(f'| {label} | {pb[key]["before_us"]:.3f} | {pb[key]["after_us"]:.3f} | {ps[key]["before_us"]:.3f} | {ps[key]["after_us"]:.3f} |')
        lines += ['',
            '当前主要剩余耗时是驻留候选/贪心和四轮交换提议；贪心自身仍约 165 μs。选择性融合恢复了 proposal/ordinal 和较大表通信的独立 kernel，只保留实测有利的融合，因此核数量多于旧的全融合路径；判断依据是完整耗时和结果，不是 kernel 数越少越好。']
    c = result['communication']; v = result['verification']
    lines += ['', '不同仪器和回放边界会改变执行时间；这些差额不能按比例从各阶段扣除，也不能拿组件微基准代替完整 planner。', '',
        '## 正确性、预算与传输计数', '',
        f'- 全部正式输出与冻结 802 原始结果逐项比较：{v["plan9_hash_comparisons"]} 次完整 plan 九项数组哈希一致；直接 CPU oracle 核验 {v["cpu_hash_comparisons"]} 次通过。没有 fallback。',
        f'- 重新核验 {v["budget_traces"]} 份预算 trace 的压缩/原始 SHA、完整回放 anchors、每次 launch 与并发线程块上限，均 ≤32，且无 NCCL kernel。这个上限不代表固定占用同一组 32 个 SM。',
        f'- 重新解析 {v["profile_trace_files"]} 份 profile trace，保留 {v["retained_profile_replays"]} 次稳态回放；{v["event_samples"]} 个 event 样本与每 rank 原始文件一致。trace 最大闭合误差 {v["trace_max_closure_us"]:.6f} μs，event 最大误差 {v["event_max_closure_us"]:.8f} μs。',
        f'- 八层累计 token-目的卡计数：原生 {c["native_remote"]:,}，新方案 {c["remote"]:,}，下降 {c["reduction_percent"]:.4f}%；专家副本累计 {c["copies"]}，与原结果相同。未测量权重/梯度/token payload、GEMM、优化器及完整训练净收益。', '',
        '原始数据位于 `final03/evidence/`。本脚本只使用标准库、冻结 parser、CPU oracle、原结果和本次冻结 manifest；输入缺失或任何核验失败时不会生成报告。运行 `python3 recompute_final32.py` 重算。']
    return '\n'.join(lines)+'\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence', type=Path, default=ROOT/'final03/evidence')
    ap.add_argument('--candidate-manifest', type=Path, default=ROOT/'candidate_manifest_03.json')
    ap.add_argument('--output-prefix', type=Path, default=ROOT/'analysis_final32')
    args = ap.parse_args()
    required = [args.evidence/(name+suffix) for name in BENCHMARKS+PROFILES for suffix in ('.json','.exit')]
    required += [args.candidate_manifest, FROZEN/'tests/profile_sampled_sm32_trace.py',
        FROZEN/'experiments/candidate_limit128/cpu_array_hashes.json', FROZEN/'previous/bounded32_03.json']
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        print(json.dumps(dict(status='waiting_for_evidence', missing=missing, analysis_written=False),indent=2))
        return 2
    manifest = load(args.candidate_manifest); sources = manifest.get('files', manifest)
    if 'payload_sha256' in manifest:
        archive = args.candidate_manifest.with_name(args.candidate_manifest.name.replace('_manifest_', '_payload_').replace('.json','.tar.gz'))
        assert archive.is_file() and sha(archive) == manifest['payload_sha256'], ('candidate source archive', archive)
    original = load(FROZEN/'previous/bounded32_03.json')
    oracle = load(FROZEN/'experiments/candidate_limit128/cpu_array_hashes.json')
    helper = FROZEN/'tests/profile_sampled_sm32_trace.py'
    spec = importlib.util.spec_from_file_location('frozen_profile_parser', helper)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    counters = dict(plan9_hash_comparisons=0, cpu_hash_comparisons=0, source_hash_comparisons=0,
                    budget_traces=0, profile_trace_files=0, profile_replays=0, retained_profile_replays=0,
                    event_samples=0, event_max_closure_us=0., trace_max_closure_us=0.)
    result = dict(status='complete', method='Formal: mean across eight layers of maximum per-rank sample median. Profile decomposition: complete median replay vector from max-median rank, per layer. Never sum independent stage maxima.',
        provenance=dict(candidate_manifest_sha256=sha(args.candidate_manifest),
            oracle_sha256=sha(FROZEN/'experiments/candidate_limit128/cpu_array_hashes.json'),
            original802_sha256=sha(FROZEN/'previous/bounded32_03.json'), parser_sha256=sha(helper)),
        verification=counters, benchmarks={}, profiles={})
    for name in BENCHMARKS:
        expected = original['metadata']['source_files'] if name.startswith('control_') else sources
        result['benchmarks'][name] = benchmark(args.evidence, name, expected, original['layers'], oracle, counters)
        print('verified',name,flush=True)
    for name in PROFILES:
        result['profiles'][name] = profile(args.evidence, name, sources, original['layers'], oracle, parser, counters)
        print('verified',name,flush=True)
    result['comparison'] = {}
    for old, new in [('bounded','bounded'),('fused','selective')]:
        before = result['benchmarks'][f'control_{old}_before03']['mean_layer_us']
        after = result['benchmarks'][f'control_{old}_after03']['mean_layer_us']
        value = result['benchmarks'][f'final_{new}03']['mean_layer_us']; midpoint = (before+after)/2
        result['comparison'][new] = dict(control_before_us=before, control_after_us=after,
            control_midpoint_us=midpoint, final_us=value, delta_us=value-midpoint,
            relative_change_percent=(value/midpoint-1)*100, speedup=midpoint/value)
    all_rows = result['benchmarks']['final_bounded03']['layers']
    native = sum(x['native_remote'] for x in all_rows); remote = sum(x['remote'] for x in all_rows)
    result['communication'] = dict(native_remote=native, remote=remote,
        reduction_percent=(1-remote/native)*100, copies=sum(x['copies'] for x in all_rows))
    for name, item in result['benchmarks'].items():
        assert [(r['remote'],r['native_remote'],r['copies']) for r in item['layers']] == [(r['remote'],r['native_remote'],r['copies']) for r in all_rows]
    for a,b in zip(result['profiles']['profile_bounded03']['layers'],result['profiles']['profile_selective03']['layers']):
        assert a['source_snapshot_hashes'] == b['source_snapshot_hashes'], ('full profile snapshot differs',a['layer'])
    assert (remote,native,result['communication']['copies']) == (3887660,5054165,434)
    old_analysis=ROOT/'analysis_baseline32.json'
    if old_analysis.is_file():
        baseline=load(old_analysis)
        assert baseline['verification']['trace_files']==256 and baseline['verification']['retained_replays']==640
        result['provenance']['baseline02_analysis_sha256']=sha(old_analysis)
        result['baseline02_phase_comparison']=compare_baseline_phases(baseline,result['profiles'])
    out = args.output_prefix.with_suffix('.json'); md = args.output_prefix.with_suffix('.md')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n'); md.write_text(markdown(result))
    print(json.dumps(dict(status='complete',json=str(out),markdown=str(md),comparison=result['comparison'],verification=counters),indent=2))
    return 0


if __name__ == '__main__': sys.exit(main())
