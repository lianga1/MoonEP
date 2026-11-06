"""Independent SM128 audit using only Python's standard library.

Run after evidence/ has been downloaded:
  python3 analysis.py

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
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT/'refs'
PLAN_FIELDS = ('dst', 'experts_to_copy', 'zero_fill_ranges', 'remote_stats',
               'cu', 'owner', 'support', 'source_info', 'history')



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
    assert len(raw) == 10 and len(record['per_rank_medians']) == 8
    medians = []
    for rank in range(8):
        for sample in raw:
            assert len(sample['per_rank']) == 8 and sample['per_rank'][rank]['rank'] == rank
            assert sample['replays'] == 5 and sample['per_rank'][rank]['cuda_ms'] > 0
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
        sample_size=256, groups_per_class=2).items()), cfg
    assert cfg['sm_limit'] in (32, 128), cfg
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
    return json.loads(raw), dict(path=str(path.resolve().relative_to(ROOT)) if ROOT in path.resolve().parents else str(path.resolve()), sha256=sha(path),
                               raw_sha256=hashlib.sha256(raw).hexdigest())


def budget_audit(data, limit):
    """Independent concurrent launched-CTA upper bound; never SM-ID occupancy."""
    kernels = [x for x in data['traceEvents'] if x.get('cat') == 'kernel' and x.get('ph') == 'X']
    assert kernels
    endpoints, blocks, streams, names = [], [], set(), []
    for item in kernels:
        args = item['args']; grid = args['grid']
        assert len(grid) == 3 and all(isinstance(x, int) and x > 0 for x in grid)
        count = math.prod(grid)
        assert count <= limit, ('grid limit', count, item['name'])
        assert 'nccl' not in item['name'].lower(), item['name']
        endpoints.extend(((item['ts'], 1, count), (item['ts']+item['dur'], 0, -count)))
        blocks.append(count); streams.add(args['stream']); names.append(item['name'])
    current = peak = 0
    for _, _, delta in sorted(endpoints):
        current += delta; peak = max(peak, current)
        assert current >= 0
    assert current == 0 and peak <= limit, ('concurrent CTA limit', peak)
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
            raw, provenance = trace_path(evidence, saved); rebuilt = budget_audit(raw, cfg['sm_limit'])
            for key, value in rebuilt.items(): assert saved[key] == value, (name, key)
            counters['budget_traces'] += 1
            traces.append(dict(layer=layer, rank=saved['rank'], **rebuilt, **provenance))
    if cfg['sm_limit'] == 128:
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
    assert data['metadata']['source_files']['tests/profile_sampled_sm32_trace.py'] == sha(FROZEN/'profile_sampled_sm32_trace.py')
    rank_raw = [load(evidence/f'{name}_rank{r}_raw.json') for r in range(8)]
    rows, all_observations = [], []
    for row in data['layers']:
        layer = row['layer']; original = originals[layer]; check = row['correctness']
        assert row['completed'] and row['fixture_sha256'] == original['fixture_sha256']
        assert cfg['sm_limit'] == 128 and cfg['fuse'] is True
        assert row['residency_frequency_sharding_enabled'] is False
        names = [x['name'] for x in row['stages']]
        assert len(names) == 36 and names[6] == 'residency_select'
        assert not any('frequency_' in name for name in names)
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
            native_grids = [math.prod(x['args']['grid']) for x, m in zip(parser.kernels(eager), mapping['launches']) if m['stage'] == 'native_seed']
            assert native_grids == [32], ('native num_sms changed', native_grids)
            frequency_names = [m['name'] for m in mapping['launches'] if 'record_frequenc' in m['name']]
            assert len(frequency_names) == 1 and 'record_frequencies_bounded' in frequency_names[0], frequency_names
            rebuilt = parser.summarize_graph(gpu_trace, mapping, 2, 5, cfg['sm_limit'])
            equal_tree(rebuilt['raw_replays'], saved['raw_replays'])
            item['trace_provenance'].append(dict(rank=rank, eager=ep, graph=gp, kernel_count_per_replay=rebuilt['kernel_count_per_replay'], native_seed_blocks=native_grids[0], local_frequency_kernel=frequency_names[0], local_frequency_blocks=sorted({x['blocks'] for r in rebuilt['raw_replays'] for x in r['launches'] if x['name']==frequency_names[0]}), max_launch_blocks=max(x['blocks'] for r in rebuilt['raw_replays'] for x in r['launches'])))
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


def markdown(result):
    b = result['benchmarks']; c = result['comparison']; a = result['profile']['aggregate']
    lines = [
        '# 现有版本128SM配置验证', '',
        '版本55564bc原样运行；同机八张H200，固定八层路由、候选128、四轮、E352/K8、全局134144个token。只测完整planner；不含token/专家权重/梯度载荷和完整训练步。', '',
        '正式口径：每卡10个样本，每样本5次Graph回放，先取每卡中位数，再取八卡最大值，最后平均同一八层。32SM前后中点用于对照，不是置信区间。', '',
        '| 配置 | 八层平均 µs | 相对32SM前后中点 |', '|---|---:|---:|',
        f"| 32SM选择性融合，前 | {b['selective32_before']['mean_layer_us']:.3f} | 对照 |",
        f"| 32SM选择性融合，后 | {b['selective32_after']['mean_layer_us']:.3f} | 对照 |",
        f"| 32SM前后中点 | {c['control_midpoint_us']:.3f} | 基准 |",
        f"| 128SM未融合 | {b['bounded128']['mean_layer_us']:.3f} | {c['bounded128']['relative_change_percent']:+.2f}% |",
        f"| 128SM选择性融合 | {b['selective128']['mean_layer_us']:.3f} | {c['selective128']['relative_change_percent']:+.2f}% |", '',
        '这里改变的是当前已提交配置：32SM自动启用八卡频率分治，128SM恢复每卡完整频率计算、少一次频率表汇总。因此不能把差异全部归因于SM数量。原生MoonEP起点仍使用32CTA；提高planner上限不会自动扩大Buffer默认num_sms。', '',
        '128SM是最多128个同时在途planner线程块所给出的SM使用上界，不表示使用了固定128个SM或达到某种利用率。', '',
        '| 配置 | 每次核数量 | 最大单次块数 | 最大同时在途块数上界 | NCCL核数 |', '|---|---:|---:|---:|---:|']
    for name in ('bounded128', 'selective128'):
        traces=b[name]['budget_traces']
        kernel_count_text=', '.join(str(value) for value in sorted(set(x['kernel_count']//3 for x in traces)))
        lines.append(f"| {name} | {kernel_count_text} | {max(x['max_launch_blocks'] for x in traces)} | {max(x['max_simultaneous_launched_blocks'] for x in traces)} | {sum(x['nccl_kernel_count'] for x in traces)} |")
    lines += ['', '## 128SM选择性融合阶段', '',
        '每层先选GPU时长中位数最大的rank，再保留该rank的一次完整中位回放，所有阶段和间隙来自同一条时间线；最后平均八层，下面各行能够闭合。不是分别挑每阶段最慢rank后相加。每组trace的前两次回放丢弃，后五次用于分析。', '',
        '| 阶段 | µs |', '|---|---:|']
    names={'entry_sync':'入口同步', 'input_copy_decode':'输入复制与解码', 'native_seed':'原生起点，混合计算/通信/等待',
        'sample_collective':'样本表合并', 'residency_local':'完整本地候选与频率/贪心', 'exchange_setup':'交换初始化',
        'exchange_propose_ordinal_prefix':'四轮提议、稳定编号与前缀', 'exchange_counts_collective':'四轮计数表收集',
        'exchange_quota':'四轮配额', 'exchange_apply':'四轮应用', 'materialize_local':'本地地址生成',
        'materialize_collective':'地址计数表收集', 'finish_local':'最终状态与方案选择',
        'finish_collective':'最终状态合并', 'publication':'来源发布与同步'}
    assert set(a['trace_group_us']) <= set(names), ('untranslated stage groups', set(a['trace_group_us'])-set(names))
    for name,label in names.items():
        if name in a['trace_group_us']:lines.append(f"| {label} | {a['trace_group_us'][name]:.3f} |")
    lines += [f"| kernel间隙 | {a['trace_gap_us']:.3f} |",f"| 完整trace span | {a['trace_span_us']:.3f} |", '',
        f"该次profile的独立无插桩完整计时前/后为{a['formal_before_us']:.3f}/{a['formal_after_us']:.3f}µs；阶段event插桩总计{a['event_total_us']:.3f}µs，相对该次无插桩前后中点增加{a['event_minus_formal_midpoint_us']:.3f}µs。Trace覆盖第一核开始到最后核结束，不能拿它替换正式Graph计时；event开销不能按固定常数分摊到各阶段。", '',
        '通信核包含信号、栅栏、等待及peer读取；原生起点也混合计算/通信/等待，没有把这些时长称为纯网络传输或纯等待。', '',
        '## 主要阶段变化', '',
        '下表32SM来自先前阶段诊断，128SM来自本次；采集批次不同，不把逐段差额当成本次正式收益的精确分摊。两边均按每层最慢卡选择一次完整中位回放，再平均八层。', '',
        '| 阶段 | 先前32SM选择性融合 µs | 本次128SM选择性融合 µs |', '|---|---:|---:|']
    old=result['previous32_stage_reference']['profile_selective03_aggregate']
    fine=lambda aggregate,fragment:sum(value for key,value in aggregate['trace_kernel_detail_us'].items() if fragment in key)
    comparisons=[('四轮提议、稳定编号与前缀',old['trace_group_us']['exchange_propose_ordinal_prefix'],a['trace_group_us']['exchange_propose_ordinal_prefix']),
        ('本地地址生成',old['trace_group_us']['materialize_local'],a['trace_group_us']['materialize_local']),
        ('候选贪心',fine(old,'global_greedy_'),fine(a,'global_greedy_')),
        ('频率计算及其额外汇总',sum(old['trace_group_us'][key] for key in ('frequency_compute','frequency_collective','frequency_unpack')),fine(a,'record_frequencies_bounded'))]
    for label,before,after in comparisons:lines.append(f'| {label} | {before:.3f} | {after:.3f} |')
    lines += ['', '当前主要节省发生在全量token扫描和地址构建。贪心仍是单块执行，没有因预算增加而加速。128SM的本地频率核实际仅启动28块；32SM分治阶段原本包含的额外汇总已经计入表中，不把它漏掉。', '', '## 计划与证据核验', '']
    v=result['verification'];traffic=result['communication']
    lines += [
        f"- 完整九项输出与冻结802计划逐项哈希核对{v['plan9_hash_comparisons']}次；CPU oracle比较{v['cpu_hash_comparisons']}次；所有物理地址/负载/Graph一致性检查通过，无fallback。",
        f"- 八层远程(token,目的卡)次数{traffic['remote']:,}，原生MoonEP为{traffic['native_remote']:,}，减少{traffic['reduction_percent']:.4f}%；专家副本累计{traffic['copies']}，四档完全一致。",
        f"- 重新解析{v['budget_traces']}份预算trace、{v['profile_trace_files']}份阶段trace，保留{v['retained_profile_replays']}次稳态回放；{v['event_samples']}个event样本。",
        '- 没有改变运行时源码；此轮验证新预算配置，不冒充重新运行全套60项测试。', '',
        '运行 `python3 analysis.py` 使用本目录evidence、refs、source_manifest.json独立复算，不需要GPU、PyTorch或私有router NPZ。摘要仅在全部输入和核验通过后写入。']
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, default=ROOT/'evidence')
    ap.add_argument('--source-manifest', type=Path, default=ROOT/'source_manifest.json')
    ap.add_argument('--output', type=Path, default=ROOT/'summary.json')
    ap.add_argument('--report', type=Path, default=ROOT/'report.md')
    args=ap.parse_args()
    names=('selective32_before','bounded128','selective128','selective32_after')
    required=[args.evidence/(name+suffix) for name in (*names,'profile128') for suffix in ('.json','.exit')]
    required += [args.source_manifest,ROOT/'run_context.json',FROZEN/'manifest.json']
    missing=[str(x) for x in required if not x.is_file()]
    if missing:
        print(json.dumps(dict(status='waiting_for_evidence',missing=missing,analysis_written=False),indent=2))
        return 2
    frozen=load(FROZEN/'manifest.json')
    for name,item in frozen.items():assert sha(FROZEN/name)==item['sha256'],('frozen reference hash',name)
    manifest=load(args.source_manifest);sources=manifest['files']
    ctx=load(ROOT/'run_context.json')
    assert ctx['commit']=='55564bcbe1febd82d99b35ae7db0e16c5ccba712'
    assert ctx['native_num_sms']==32 and ctx['frequency_sharding']=={'32':True,'128':False}
    original=load(FROZEN/'original802_bounded32.json');oracle=load(FROZEN/'cpu_array_hashes.json')
    helper=FROZEN/'profile_sampled_sm32_trace.py'
    assert sources['tests/profile_sampled_sm32_trace.py']==sha(helper)
    spec=importlib.util.spec_from_file_location('frozen_parser',helper)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    counters=dict(plan9_hash_comparisons=0,cpu_hash_comparisons=0,source_hash_comparisons=0,budget_traces=0,
        profile_trace_files=0,profile_replays=0,retained_profile_replays=0,event_samples=0,
        event_max_closure_us=0.,trace_max_closure_us=0.)
    result=dict(status='complete',method='Per-layer maximum of eight rank sample medians, then mean of same eight layers; stages use one complete critical-rank median replay vector.',
        provenance=dict(commit=ctx['commit'],source_manifest_sha256=sha(args.source_manifest),
            refs_manifest_sha256=sha(FROZEN/'manifest.json'),parser_sha256=sha(helper)),
        verification=counters,benchmarks={})
    for name in names:
        item=benchmark(args.evidence,name,sources,original['layers'],oracle,counters)
        assert item['config']['sm_limit']==(32 if name.startswith('selective32') else 128)
        assert item['config']['fuse']==(name!='bounded128')
        result['benchmarks'][name]=item
        print('verified',name,flush=True)
    hosts=[]
    for item in result['benchmarks'].values():
        md=item['metadata'];hosts.append(sorted((x['rank'],x['host'],x['uuid']) for x in md['identities']))
        assert md['device_sms']==132 and 'H200' in md['gpu']
        assert md['samples']==10 and md['replays']==5
    assert all(host==hosts[0] for host in hosts)
    result['profile']=profile(args.evidence,'profile128',sources,original['layers'],oracle,parser,counters)
    profile_md=result['profile']['metadata']
    profile_devices=profile_md.get('identities',profile_md.get('devices'))
    assert sorted((x['rank'],x['host'],x['uuid']) for x in profile_devices)==hosts[0]
    result['profile']['observed_native_seed_blocks']=sorted({x['native_seed_blocks'] for layer in result['profile']['layers'] for x in layer['trace_provenance']})
    result['profile']['observed_local_frequency_blocks']=sorted({v for layer in result['profile']['layers'] for x in layer['trace_provenance'] for v in x['local_frequency_blocks']})
    result['profile']['observed_max_launch_blocks']=max(x['max_launch_blocks'] for layer in result['profile']['layers'] for x in layer['trace_provenance'])
    print('verified profile128',flush=True)
    rows=result['benchmarks']['selective128']['layers']
    for item in result['benchmarks'].values():
        assert [(r['remote'],r['native_remote'],r['copies']) for r in item['layers']]==[(r['remote'],r['native_remote'],r['copies']) for r in rows]
    remote=sum(x['remote'] for x in rows);native=sum(x['native_remote'] for x in rows);copies=sum(x['copies'] for x in rows)
    assert (remote,native,copies)==(3887660,5054165,434)
    result['communication']=dict(remote=remote,native_remote=native,reduction_percent=(1-remote/native)*100,copies=copies)
    b=result['benchmarks'];before=b['selective32_before']['mean_layer_us'];after=b['selective32_after']['mean_layer_us'];mid=(before+after)/2
    result['comparison']=dict(control_before_us=before,control_after_us=after,control_midpoint_us=mid,
        control_drift_percent=(after/before-1)*100)
    for name in ('bounded128','selective128'):
        value=b[name]['mean_layer_us']
        result['comparison'][name]=dict(us=value,delta_us=value-mid,relative_change_percent=(value/mid-1)*100,speedup=mid/value)
    result['comparison']['fusion_within128']=dict(delta_us=b['selective128']['mean_layer_us']-b['bounded128']['mean_layer_us'],
        relative_change_percent=(b['selective128']['mean_layer_us']/b['bounded128']['mean_layer_us']-1)*100)
    previous=load(FROZEN/'previous32_profile_summary.json')
    result['previous32_stage_reference']=dict(warning='Previous experiment cohort, not current paired timing; stage differences are diagnostic.',**previous)
    args.output.write_text(json.dumps(result,indent=2)+'\n');args.report.write_text(markdown(result))
    print(json.dumps(dict(status='complete',output=str(args.output),report=str(args.report),comparison=result['comparison'],verification=counters),indent=2))
    return 0


if __name__=='__main__':sys.exit(main())
