from pathlib import Path
import json,time,hashlib
import numpy as np
from vendor.aggregate_exchange import optimize_aggregate_multipair as optimize_aggregate,match_counts
from moonep_cpu import communication
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/aggregate_exchange_multipair';OUT.mkdir(exist_ok=True)
# Required case: one size-3 group can be balanced by a size-1 + size-2 group.
h=np.zeros((2,2,2,4),np.int64);h[0,1,1,3]=1;h[1,0,0,1]=1;h[1,0,0,2]=1
ac,pred,_,_=match_counts(h,2,3,True)
assert pred==1 and ac[0,1,1,3]==1 and ac[1,0,0,1]==1 and ac[1,0,0,2]==1
# Compile all paths before real CPU observations; these are not GPU timings.
x=np.array([[[0,2],[1,3]],[[0,3],[1,2]]],np.int32);o=x//2;su=np.ones((2,4),bool)
for mix in (False,True):optimize_aggregate(x,o,su,1,mix)
rows=[]
for layer in range(8):
 inp=ROOT/f'outputs/trace_audit/layer_{layer:02d}.npz';x=np.load(inp)['topk'].astype(np.int32)
 seed=np.load(ROOT/f'outputs/rt2_b8_comparison/layer{layer:02d}_native_rt0.npz')['owner']
 saved=ROOT/f'outputs/replay/layer{layer:02d}_b8_small256_global_4.npz';su=np.load(saved)['support']
 base=communication(x,seed,352);row={'layer':layer,'baseline':base['remote_token_destinations'],'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'support_sha256':hashlib.sha256(su.tobytes()).hexdigest(),'variants':{}}
 for mix in (False,True):
  name='mixed_sizes' if mix else 'equal_sizes';start=time.perf_counter();dst,history=optimize_aggregate(x,seed,su,4,mix);elapsed=time.perf_counter()-start
  m=communication(x,dst,352);extras=(su&(np.arange(352)[None,:]//44!=np.arange(8)[:,None])).sum(1)
  assert max(extras)<=8 and np.all(su[dst,x]);assert m['pair_loads']==base['pair_loads']
  assert base['remote_token_destinations']-m['remote_token_destinations']==sum(r['predicted_gain'] for r in history)
  np.savez_compressed(OUT/f'layer{layer:02d}_{name}.npz',owner=dst,support=su)
  row['variants'][name]={'metrics':m,'history':history,'cpu_seconds':elapsed,'reduction_pct':100*(1-m['remote_token_destinations']/base['remote_token_destinations']),'audit_pass':True}
  print(layer,name,row['variants'][name]['reduction_pct'],'states',[r['unique_states'] for r in history],flush=True)
 rows.append(row);(OUT/'results.json').write_text(json.dumps({'scope':'CPU semantic prototype; fixed saved 256/g2 residency; no GPU benchmark','rows':rows},indent=2))
print('DONE',flush=True)
