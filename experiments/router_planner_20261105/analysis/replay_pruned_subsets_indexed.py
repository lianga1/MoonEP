from pathlib import Path
import json,time,hashlib
import numpy as np
from vendor.pruned_subset_exchange import prepare,execute
from moonep_cpu import communication
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/pruned_subset_indexed';OUT.mkdir(exist_ok=True)
# Warm compile, not counted as latency.
x=np.array([[[0,2],[1,3]],[[0,3],[1,2]]],np.int32);o=x//2;su=np.ones((2,4),bool)
for h in (1,2):p,_=prepare(x,o,su,h,2,True);execute(x,o,su,p,1)
rows=[]
for layer in range(8):
 inp=ROOT/f'outputs/trace_audit/layer_{layer:02d}.npz';x=np.load(inp)['topk'].astype(np.int32)
 seed=np.load(ROOT/f'outputs/rt2_b8_comparison/layer{layer:02d}_native_rt0.npz')['owner']
 su=np.load(ROOT/f'outputs/replay/layer{layer:02d}_b8_small256_global_4.npz')['support']
 base=communication(x,seed,352);row={'layer':layer,'baseline':base['remote_token_destinations'],'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'variants':{}}
 for h,g in ((1,8),(2,2),(2,8),(2,32),(2,0)):
  name=f'h{h}_g{g}';t=time.perf_counter();p,diag=prepare(x,seed,su,h,g,True);prep=time.perf_counter()-t
  times=[]
  for repeat in range(3):
   t=time.perf_counter();dst,history=execute(x,seed,su,p,4);times.append(time.perf_counter()-t)
  m=communication(x,dst,352)
  assert m['pair_loads']==base['pair_loads'] and np.all(su[dst,x])
  assert base['remote_token_destinations']-m['remote_token_destinations']==sum(v['predicted_gain'] for v in history)
  assert max((su&(np.arange(352)[None,:]//44!=np.arange(8)[:,None])).sum(1))<=8
  np.savez_compressed(OUT/f'layer{layer:02d}_{name}.npz',owner=dst,support=su)
  row['variants'][name]=dict(metrics=m,history=history,diagnostics=diag,cpu_prepare_seconds=prep,cpu_exchange_seconds=times,cpu_total_estimate_seconds=prep+float(np.median(times)),reduction_pct=100*(1-m['remote_token_destinations']/base['remote_token_destinations']),audit_pass=True)
  print(layer,name,round(row['variants'][name]['reduction_pct'],4),'patterns',diag['selected_direction_patterns'],'prep',round(prep,3),'exchange',round(np.median(times),3),flush=True)
 rows.append(row);(OUT/'results.json').write_text(json.dumps({'scope':'CPU NumPy/Numba semantic and latency prototype; prepare measured once; exchange 3 repeats; excludes residency, seed, final address encoding, GPU execution','rows':rows},indent=2))
