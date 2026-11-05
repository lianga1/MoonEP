from pathlib import Path
from itertools import product
import json,numpy as np
from vendor.aggregate_exchange import bounded_dp,match_counts
root=Path(__file__).resolve().parents[1];rng=np.random.default_rng(20261102)
# Exhaustive small-count reference for the mixed-size matcher.
for case in range(40):
 c=rng.integers(0,3,size=(2,4),dtype=np.int64);c[:,0]=0;cap=12
 got,_=bounded_dp(c,cap,3);expected=np.full(cap+1,-100000000,dtype=np.int32)
 for ns in product(*(range(int(c[g,s])+1) for g in range(2) for s in range(1,4))):
  weight=sum(ns[i]*(i%3+1) for i in range(6));gain=sum(ns[3:])
  if weight<=cap:expected[weight]=max(expected[weight],gain)
 assert np.array_equal(got,expected)
checks=[]
for folder in ('aggregate_exchange','aggregate_exchange_multipair'):
 rows=json.loads((root/'outputs'/folder/'results.json').read_text())['rows'];assert len(rows)==8
 for row in rows:
  l=row['layer'];x=np.load(root/f'outputs/trace_audit/layer_{l:02d}.npz')['topk'];src=np.arange(8)[:,None,None]
  original=np.load(root/f'outputs/replay/layer{l:02d}_b8_small256_global_4.npz')['support']
  for mode,v in row['variants'].items():
   z=np.load(root/f'outputs/{folder}/layer{l:02d}_{mode}.npz');o=z['owner'];su=z['support']
   assert np.array_equal(su,original);assert o.shape==x.shape and ((o>=0)&(o<8)).all();assert su[o,x].all()
   assert np.array_equal(np.bincount(o.ravel(),minlength=8),np.full(8,134144))
   assert max((su&(np.arange(352)[None,:]//44!=np.arange(8)[:,None])).sum(1))<=8
   ordered=np.sort(o,axis=-1);first=np.concatenate([np.ones((8,16768,1),bool),ordered[:,:,1:]!=ordered[:,:,:-1]],axis=-1)
   remote=int((first&(ordered!=src)).sum());assert remote==v['metrics']['remote_token_destinations']
   assert row['baseline']-remote==sum(h['predicted_gain'] for h in v['history'])
   checks.append(dict(folder=folder,layer=l,mode=mode,remote=remote,pass_all=True))
summary={}
for folder in ('aggregate_exchange','aggregate_exchange_multipair'):
 rows=json.loads((root/'outputs'/folder/'results.json').read_text())['rows'];base=sum(r['baseline'] for r in rows)
 for mode in ('equal_sizes','mixed_sizes'):
  rem=sum(r['variants'][mode]['metrics']['remote_token_destinations'] for r in rows)
  unique=sum(sum(h['unique_states'] for h in r['variants'][mode]['history']) for r in rows)
  summary[folder+'_'+mode]=dict(remote=rem,reduction_pct=100*(1-rem/base),mean_cpu_seconds=np.mean([r['variants'][mode]['cpu_seconds'] for r in rows]),representative_scans=unique,original_token_rounds=8*4*134144)
(root/'outputs/aggregate_exchange_multipair/independent_audit.json').write_text(json.dumps({'dp_exhaustive_cases':40,'saved_plan_checks':checks,'summary':summary},indent=2))
print(json.dumps(summary,indent=2))
