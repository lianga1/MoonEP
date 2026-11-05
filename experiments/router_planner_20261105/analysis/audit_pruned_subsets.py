from pathlib import Path
import json,numpy as np
from vendor.pruned_subset_exchange import mine
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/pruned_subset_indexed'
# Identical expert subset has different eligibility under different other owners.
x=np.array([[3,4,5],[3,4,5],[3,4,5],[3,4,5],[0,1,2],[0,1,2]],np.int32)
o=np.array([[0,0,0],[0,0,0],[1,1,1],[1,1,1],[0,0,0],[0,0,1]],np.int32)
su=np.ones((3,9),bool);keys,vals,_=mine(x,o,su,3,2,9,2)
base=9+81;pattern=9+0*9+1
key=((((0*3+1)*2+1)*2+1)*base)+pattern
assert dict(zip(keys,vals))[key]==1
negative_zero=((((0*3+1)*2+0)*2+1)*base)+pattern
assert negative_zero not in dict(zip(keys,vals))
rows=json.loads((OUT/'results.json').read_text())['rows'];assert len(rows)==8;checks=[]
for row in rows:
 l=row['layer'];x=np.load(ROOT/f'outputs/trace_audit/layer_{l:02d}.npz')['topk'];orig=np.load(ROOT/f'outputs/replay/layer{l:02d}_b8_small256_global_4.npz')['support']
 for name,v in row['variants'].items():
  z=np.load(OUT/f'layer{l:02d}_{name}.npz');o=z['owner'];su=z['support']
  assert np.array_equal(su,orig) and o.shape==x.shape and ((o>=0)&(o<8)).all() and su[o,x].all()
  assert max((su&(np.arange(352)[None,:]//44!=np.arange(8)[:,None])).sum(1))<=8
  assert np.array_equal(np.bincount(o.ravel(),minlength=8),np.full(8,134144))
  so=np.sort(o,axis=-1);first=np.concatenate([np.ones((8,16768,1),bool),so[:,:,1:]!=so[:,:,:-1]],axis=-1)
  remote=int((first&(so!=np.arange(8)[:,None,None])).sum())
  assert remote==v['metrics']['remote_token_destinations']
  assert row['baseline']-remote==sum(h['predicted_gain'] for h in v['history'])
  unfiltered=np.load(ROOT/f'outputs/pruned_subset_exchange/layer{l:02d}_{name}.npz')['owner']
  assert np.array_equal(o,unfiltered),'filtered index changed decisions'
  checks.append(dict(layer=l,variant=name,all_pass=True,remote=remote))
summary={};bench=json.loads((OUT/'cpu_benchmark.json').read_text());assert len(bench['rows'])==8
for name in rows[0]['variants']:
 vv=[r['variants'][name] for r in rows];base=sum(r['baseline'] for r in rows);remote=sum(v['metrics']['remote_token_destinations'] for v in vv)
 summary[name]=dict(remote=remote,reduction_pct=100*(1-remote/base),pattern_range=[min(v['diagnostics']['selected_direction_patterns'] for v in vv),max(v['diagnostics']['selected_direction_patterns'] for v in vv)],mean_prepare_ms=1000*np.mean([v['cpu_prepare_seconds'] for v in vv]),mean_exchange_median_ms=1000*np.mean([np.median(v['cpu_exchange_seconds']) for v in vv]),mean_posting_entries=np.mean([v['diagnostics']['posting_entries'] for v in vv]),mean_four_round_posting_visits=np.mean([sum(h['posting_visits'] for h in v['history']) for v in vv]))
timing={name:1000*np.mean([np.median(r['seconds'][name]) for r in bench['rows']]) for name in bench['rows'][0]['seconds']}
(OUT/'summary.json').write_text(json.dumps(dict(quality=summary,paired_cpu_layer_median_mean_ms=timing,checks=checks,context_sensitive_subset_test=True),indent=2))
print(json.dumps(dict(quality=summary,timing=timing),indent=2))
