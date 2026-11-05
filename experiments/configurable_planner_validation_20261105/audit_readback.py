"""Independently read retained logs/results and bind them to published source."""
from pathlib import Path
import argparse, hashlib, json, re, statistics
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--previous-benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists(),a.output
checks={}
def require(ok,label):
 checks[label]=bool(ok)
 assert ok,label
try:
 exits={name:int((a.evidence_root/f'{name}.exit').read_text()) for name in ('host_01','matrix_02','default_01','replay_01','residency_01','weighted_01','benchmark_01','compile_01','compile_remaining_01')}
 require(all(v==0 for v in exits.values()),'all_completed_commands_exit_zero')
 counts={}
 for name,number,ranks in [('host_01',13,1),('matrix_02',14,8),('default_01',6,8),('residency_01',4,1),('weighted_01',2,8)]:
  text=(a.evidence_root/f'{name}.log').read_text()
  count=len(re.findall(r'(?<!\d)'+str(number)+r' passed',text))
  counts[name]=dict(tests_per_process=number,processes_reported=count)
  require(count==ranks,f'{name}_all_processes_pass')
 require(int((a.evidence_root/'matrix_01.exit').read_text())!=0,'partial_attempt_preserved')
 require((a.evidence_root/'matrix_01.stop_reason').exists(),'partial_attempt_has_explicit_reason')
 manifest=json.loads((a.evidence_root/'runtime_manifest_final.json').read_text())
 for path,digest in manifest['source_sha256'].items():
  require(hashlib.sha256((a.source_root/path).read_bytes()).hexdigest()==digest,f'source:{path}')
 replay=json.loads((a.evidence_root/'replay_01.json').read_text())
 require(replay['status']=='complete' and not replay['failures'],'component_replay_complete')
 for key in ('all_eight_layers_checked','all_original_source_round_owners_exact','original_support_and_copies_exact','graph_dynamic_layers_checked'):require(replay[key],key)
 benchmark=json.loads((a.evidence_root/'benchmark_01.json').read_text())
 old=json.loads(a.previous_benchmark.read_text())
 require(benchmark['status']=='complete' and not benchmark['failures'],'eight_gpu_benchmark_complete')
 require(len(benchmark['layers'])==len(old['layers'])==8,'all_eight_original_layers')
 totals={v:dict(remote=0,copies=0,sum_layer_ms=0.) for v in ('moon','sampled_group')}
 for layer,prior in zip(benchmark['layers'],old['layers']):
  i=layer['layer'];require(i==prior['layer'],f'layer{i}_identity')
  require(layer['fixture']==prior['fixture'],f'layer{i}_same_original_fixture')
  require(layer['post_timing_audit_passed'] and layer['native_seed_matches_direct_baseline'] and layer['runtime_routes_match_local_fixture'],f'layer{i}_all_runtime_audits')
  require(layer['planner_status_by_rank']==[0]*8,f'layer{i}_no_fallback')
  for v,kind in [('moon','native'),('sampled_group','sampled_group')]:
   variant=layer['variants'][v]
   samples=variant['raw'];require(len(samples)==10 and all(x['replays']==5 for x in samples),f'layer{i}_{v}_fixed_timing_protocol')
   medians=[statistics.median(sample['per_rank'][r]['cuda_ms'] for sample in samples) for r in range(8)]
   measured=max(medians)
   require(abs(measured-variant['max_rank_median_cuda_ms'])<1e-9,f'layer{i}_{v}_recomputed_time')
   audit=layer['audit'][kind];prev=prior['audit'][kind]
   require(audit['passed'] and all(audit['checks'].values()),f'layer{i}_{v}_physical_checks')
   require(audit['owner_sha256']==prev['owner_sha256'],f'layer{i}_{v}_same_owner_as_previous_commit')
   require(audit['expert_copies']==prev['expert_copies'],f'layer{i}_{v}_same_copy_count')
   require(sum(map(sum,audit['dispatch_matrix']))==audit['remote_token_destinations'],f'layer{i}_{v}_communication_count')
   require(sum(map(sum,audit['home_to_replica_matrix']))==audit['expert_copies'],f'layer{i}_{v}_copy_matrix_count')
   totals[v]['remote']+=audit['remote_token_destinations'];totals[v]['copies']+=audit['expert_copies'];totals[v]['sum_layer_ms']+=measured
 for v,key in [('moon','native_planner_sum_layer_ms'),('sampled_group','sampled_planner_sum_layer_ms')]:
  require(abs(totals[v]['sum_layer_ms']-benchmark['summary'][key])<1e-9,f'{v}_summary_time')
  totals[v]['mean_layer_us']=totals[v]['sum_layer_ms']/8*1000
 result=dict(status='complete',checks=checks,exit_codes=exits,pytest_counts=counts,totals=totals,remote_reduction_percent=100*(1-totals['sampled_group']['remote']/totals['moon']['remote']),previous_commit_mean_layer_us={v:old['summary'][key]/8*1000 for v,key in [('moon','native_planner_sum_layer_ms'),('sampled_group','sampled_planner_sum_layer_ms')]},scope='planner-only timing; no end-to-end training speedup claim')
except Exception as exc:
 result=dict(status='failed',checks=checks,error=f'{type(exc).__name__}: {exc}')
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 raise
else:
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
