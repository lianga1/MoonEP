"""Independent raw-data recomputation for baseline_bounded/fused32_02.

Run here with Python's standard library; no CUDA/PyTorch is required.
"""
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT/'baseline_profiles_02/evidence'
SOURCE = ROOT/'baseline_profiles_02/analysis_sources'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol=1e-5):
    assert abs(a-b) <= tol, (a, b, a-b)


def equal_tree(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    elif isinstance(a,float): close(a,b,1e-8)
    else: assert a==b,(a,b)


def mean_dict(values):
    keys = set().union(*(set(x) for x in values))
    return {key: statistics.mean(x.get(key, 0.) for x in values) for key in sorted(keys)}


def stage_group(stage):
    if stage == 'native_seed': return 'native_seed'
    if stage == 'pre_sync': return 'entry_sync'
    if stage in ('input_copy', 'decode_owner'): return 'input_copy_decode'
    if stage.startswith('residency_'):
        return 'sample_collective' if 'collective' in stage else 'residency_local'
    if stage.startswith('exchange_'): return 'exchange_setup'
    if stage.startswith('round'):
        return 'exchange_'+stage.split('_', 1)[1]
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


def count_hashes(value):
    if isinstance(value, dict):
        if 'sha256' in value: return 1
        return sum(count_hashes(x) for x in value.values())
    return 0


def formal(raw):
    ranks = range(8)
    medians = [statistics.median(x['per_rank'][r]['cuda_ms'] for x in raw['raw'])*1000 for r in ranks]
    close(max(medians), raw['max_rank_median_cuda_ms']*1000)
    return dict(max_rank_median_us=max(medians), rank_median_us=medians)


def critical_vector(ranks, key):
    """Pick max rank median, retain its one/two middle complete observations."""
    medians = [statistics.median(x[key] for x in rows) for rows in ranks]
    rank = max(range(len(ranks)), key=lambda r: medians[r])
    ordered = sorted(ranks[rank], key=lambda x: x[key])
    middle = [ordered[len(ordered)//2]] if len(ordered)%2 else ordered[len(ordered)//2-1:len(ordered)//2+1]
    out = dict(rank=rank, observations=[x['replay'] for x in middle], total_us=medians[rank])
    for name in ('stage_us', 'group_us', 'category_us', 'kernel_us'):
        if name in middle[0]: out[name] = mean_dict([x[name] for x in middle])
    for name in ('gap_us', 'kernel_sum_us'):
        if name in middle[0]: out[name] = statistics.mean(x[name] for x in middle)
    return out


def main():
    helper = SOURCE/'tests/profile_sampled_sm32_trace.py'
    spec = importlib.util.spec_from_file_location('trace_parser', helper)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    records = {}
    verification = dict(trace_files=0, trace_replays=0, retained_replays=0,
                        stage_samples=0, raw_rank_files=0, trace_max_closure_us=0., event_max_closure_us=0.,
                        output_hash_comparisons_between_variants=0, previous_hash_comparisons=0,
                        direct_cpu_oracle_hash_comparisons=0)
    cpu_path=SOURCE/'experiments/candidate_limit128/cpu_array_hashes.json'
    cpu_oracle=json.loads(cpu_path.read_text())
    verification['cpu_oracle_sha256']=sha(cpu_path)
    previous_dir = SOURCE/'previous'
    previous = {variant: json.loads((previous_dir/f'{variant}32_03.json').read_text())
                for variant in ('bounded', 'fused')}
    for variant in ('bounded', 'fused'):
        path = EVIDENCE/f'baseline_{variant}32_02.json'
        data = json.loads(path.read_text())
        assert data['status'] == 'complete' and not data['failures']
        assert data['metadata']['config']['sm_limit'] == 32
        assert data['metadata']['config']['fuse'] == (variant == 'fused')
        for name in ('tests/profile_sampled_sm32.py', 'tests/profile_sampled_sm32_trace.py'):
            assert sha(SOURCE/name) == data['metadata']['source_files'][name]
        rank_raw = [json.loads((EVIDENCE/f'baseline_{variant}32_02_rank{r}_raw.json').read_text()) for r in range(8)]
        verification['raw_rank_files'] += len(rank_raw)
        result = dict(input_sha256=sha(path), source_files=data['metadata']['source_files'], layers=[],
                      metadata=data['metadata'])
        records[variant] = result
        assert [x['layer'] for x in data['layers']] == list(range(8))
        all_trace_rows = []
        for layer, row in enumerate(data['layers']):
            assert row['completed']
            checks = row['correctness']
            for name in ('native', 'sampled'):
                assert checks[name]['passed'] and all(checks[name]['checks'].values())
            assert checks['sampled']['status_by_rank'] == [0]*8 and not checks['sampled']['fell_back_to_native']
            for name in ('split_eager_exact', 'external_event_graph_exact', 'steady_profiler_graph_exact', 'formal_after_graph_exact'):
                assert len(checks[name]) == 8 and all(not x['mismatches'] for x in checks[name])
            original = previous[variant]['layers'][layer]
            assert original['fixture_sha256'] == row['fixture_sha256']
            assert original['cpu_oracle_checked']
            cpu=cpu_oracle[f'layer_{layer:02d}_cap128']
            assert checks['sampled']['owner_sha256']==cpu['owner_rounds4']['sha256']
            verification['direct_cpu_oracle_hash_comparisons']+=1
            for r, old in enumerate(original['output_hashes_by_rank']):
                for field in ('support','history'):
                    assert checks['production_output_hashes_by_rank'][r][field] == cpu[field]
                    verification['direct_cpu_oracle_hash_comparisons']+=1
                for field, expected in old.items():
                    assert checks['production_output_hashes_by_rank'][r][field]['sha256'] == expected
                    verification['previous_hash_comparisons'] += 1
            item = dict(layer=layer, formal_before=formal(row['formal_before']),
                formal_after=formal(row['formal_after']), formal_native=formal(row['formal_native']),
                native_remote=checks['native']['remote_token_destinations'],
                sampled_remote=checks['sampled']['remote_token_destinations'],
                copies=checks['sampled']['expert_copies'], trace_by_rank=[], stage_events_by_rank=[],
                output_hashes=checks['production_output_hashes_by_rank'])
            result['layers'].append(item)
            for rank in range(8):
                assert rank_raw[rank]['layers'][layer]['stage_events'] == row['stage_events_by_rank'][rank]
                assert rank_raw[rank]['layers'][layer]['trace'] == row['traces_by_rank'][rank]
                er = []
                for event in row['stage_events_by_rank'][rank]['raw']:
                    total = sum(event['stage_us'].values())
                    close(total, event['sum_stage_us'], 1e-9)
                    error = event['total_us']-total
                    close(error, event['closure_error_us'], 1e-9)
                    verification['event_max_closure_us'] = max(verification['event_max_closure_us'], abs(error))
                    grouped = defaultdict(float)
                    for stage, duration in event['stage_us'].items(): grouped[stage_group(stage)] += duration
                    er.append(dict(replay=event['sample'], total_us=event['total_us'], stage_us=event['stage_us'], group_us=dict(grouped)))
                    verification['stage_samples'] += 1
                item['stage_events_by_rank'].append(er)
                saved = row['traces_by_rank'][rank]
                loaded = {}
                for field in ('eager_trace', 'graph_trace'):
                    desc = saved[field]
                    p = EVIDENCE/Path(desc['path']).parent.name/Path(desc['path']).name
                    assert sha(p) == desc['sha256']
                    loaded[field] = parser.load_trace(p)
                    verification['trace_files'] += 1
                mapping = parser.eager_mapping(loaded['eager_trace'], row['stages'])
                assert mapping == saved['mapping']
                # Re-run the pinned parser, then independently sum the raw
                # events below. Eager durations never enter the estimates.
                rebuilt = parser.summarize_graph(loaded['graph_trace'], mapping, 2, 5, 32)
                equal_tree(rebuilt['raw_replays'], saved['raw_replays'])
                gpu = parser.kernels(loaded['graph_trace'])
                n = len(mapping['launches'])
                assert n == (104 if variant == 'bounded' else 64) and len(gpu) == 7*n
                durations = []
                for replay in range(7):
                    chunk = gpu[replay*n:(replay+1)*n]
                    stages, categories, fine = defaultdict(float), defaultdict(float), defaultdict(float)
                    for k, info in zip(chunk, mapping['launches']):
                        stages[info['stage']] += k['dur']; categories[info['category']] += k['dur']
                        fine[kernel_key(info['stage'], k['name'])] += k['dur']
                    grouped = defaultdict(float)
                    for stage, duration in stages.items(): grouped[stage_group(stage)] += duration
                    gaps = sum(max(0., chunk[i]['ts']-(chunk[i-1]['ts']+chunk[i-1]['dur'])) for i in range(1,n))
                    total = chunk[-1]['ts']+chunk[-1]['dur']-chunk[0]['ts']
                    kernels = sum(x['dur'] for x in chunk)
                    error = total-kernels-gaps
                    verification['trace_max_closure_us'] = max(verification['trace_max_closure_us'], abs(error))
                    close(error, 0, .01)
                    value = dict(replay=replay, total_us=total, kernel_sum_us=kernels, gap_us=gaps,
                                 stage_us=dict(stages), group_us=dict(grouped), category_us=dict(categories), kernel_us=dict(fine))
                    verification['trace_replays'] += 1
                    if replay >= 2:
                        durations.append(value); all_trace_rows.append(value); verification['retained_replays'] += 1
                item['trace_by_rank'].append(durations)
            item['critical_trace'] = critical_vector(item['trace_by_rank'], 'total_us')
            item['critical_events'] = critical_vector(item['stage_events_by_rank'], 'total_us')
            close(sum(item['critical_trace']['group_us'].values())+item['critical_trace']['gap_us'], item['critical_trace']['total_us'], .01)
            print(variant, layer, 'formal', item['formal_before']['max_rank_median_us'], 'trace', item['critical_trace']['total_us'], flush=True)
        result['aggregate'] = dict(
            formal_before_us=statistics.mean(x['formal_before']['max_rank_median_us'] for x in result['layers']),
            formal_after_us=statistics.mean(x['formal_after']['max_rank_median_us'] for x in result['layers']),
            formal_native_us=statistics.mean(x['formal_native']['max_rank_median_us'] for x in result['layers']),
            trace_span_us=statistics.mean(x['critical_trace']['total_us'] for x in result['layers']),
            trace_kernel_us=statistics.mean(x['critical_trace']['kernel_sum_us'] for x in result['layers']),
            trace_gap_us=statistics.mean(x['critical_trace']['gap_us'] for x in result['layers']),
            event_total_us=statistics.mean(x['critical_events']['total_us'] for x in result['layers']),
            trace_group_us=mean_dict([x['critical_trace']['group_us'] for x in result['layers']]),
            trace_stage_us=mean_dict([x['critical_trace']['stage_us'] for x in result['layers']]),
            trace_category_us=mean_dict([x['critical_trace']['category_us'] for x in result['layers']]),
            trace_kernel_detail_us=mean_dict([x['critical_trace']['kernel_us'] for x in result['layers']]),
            event_group_us=mean_dict([x['critical_events']['group_us'] for x in result['layers']]),
            all_rank_replay_mean_span_us=statistics.mean(x['total_us'] for x in all_trace_rows),
            all_rank_replay_mean_gap_us=statistics.mean(x['gap_us'] for x in all_trace_rows),
            all_rank_replay_mean_group_us=mean_dict([x['group_us'] for x in all_trace_rows]),
            all_rank_replay_mean_kernel_us=mean_dict([x['kernel_us'] for x in all_trace_rows]))
        agg = result['aggregate']
        agg['formal_midpoint_us'] = (agg['formal_before_us']+agg['formal_after_us'])/2
        agg['event_minus_formal_midpoint_us'] = agg['event_total_us']-agg['formal_midpoint_us']
        agg['trace_minus_formal_midpoint_us'] = agg['trace_span_us']-agg['formal_midpoint_us']
    for a,b in zip(records['bounded']['layers'], records['fused']['layers']):
        assert a['output_hashes'] == b['output_hashes']
        verification['output_hash_comparisons_between_variants'] += sum(count_hashes(x) for x in a['output_hashes'])
        assert a['native_remote'] == b['native_remote'] and a['sampled_remote'] == b['sampled_remote'] and a['copies'] == b['copies']
    b, f = [records[v]['aggregate'] for v in ('bounded','fused')]
    comparison = dict(formal_midpoint_delta_us=f['formal_midpoint_us']-b['formal_midpoint_us'],
        trace_span_delta_us=f['trace_span_us']-b['trace_span_us'],
        trace_kernel_delta_us=f['trace_kernel_us']-b['trace_kernel_us'],
        trace_gap_delta_us=f['trace_gap_us']-b['trace_gap_us'])
    for field in ('trace_group_us', 'trace_stage_us', 'trace_category_us', 'trace_kernel_detail_us', 'all_rank_replay_mean_group_us'):
        comparison[field+'_delta']={key:f[field].get(key,0.)-b[field].get(key,0.) for key in sorted(set(b[field])|set(f[field]))}
    result = dict(method='Formal=max over rank medians per layer, then eight-layer mean. Trace decomposition=choose rank with max median span per layer; take complete median replay vector, then average eight layers. Event decomposition=likewise average complete central two observations on chosen rank. No per-stage max sum.',
        verification=verification, variants=records, comparison=comparison)
    (ROOT/'analysis_baseline32.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verification=verification,bounded=b,fused=f,comparison=comparison),indent=2))


if __name__ == '__main__': main()
