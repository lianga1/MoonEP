"""Paired rotating-order local CPU timings. Excludes JIT and disk I/O.
Includes preparation + all four exchange rounds for every timed candidate.
Fixed native seed and residency are inputs to all methods.
"""
from pathlib import Path
import json,time,platform,sys
import numpy as np
import numba
from vendor.pruned_subset_exchange import prepare,execute
from vendor.aggregate_exchange import optimize_aggregate_multipair
from vendor.bucket_optimizer import optimize
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/pruned_subset_indexed';OUT.mkdir(exist_ok=True)
methods=['token_bundle','aggregate_mixed','subset_h1_g8','subset_h2_g2','subset_h2_g8','subset_h2_all']
def run(name,x,o,su):
 if name=='token_bundle':return optimize(x,o,su,rounds=4,mode='bundle')[0]
 if name=='aggregate_mixed':return optimize_aggregate_multipair(x,o,su,rounds=4,mixed=True)[0]
 h,g={'subset_h1_g8':(1,8),'subset_h2_g2':(2,2),'subset_h2_g8':(2,8),'subset_h2_all':(2,0)}[name]
 p,_=prepare(x,o,su,h,g,True);return execute(x,o,su,p,4)[0]
x=np.array([[[0,2],[1,3]],[[0,3],[1,2]]],np.int32);o=x//2;su=np.ones((2,4),bool)
for name in methods:run(name,x,o,su)
rows=[]
for l in range(8):
 x=np.load(ROOT/f'outputs/trace_audit/layer_{l:02d}.npz')['topk'].astype(np.int32)
 seed=np.load(ROOT/f'outputs/rt2_b8_comparison/layer{l:02d}_native_rt0.npz')['owner']
 saved=np.load(ROOT/f'outputs/replay/layer{l:02d}_b8_small256_global_4.npz');su=saved['support']
 expected={'token_bundle':saved['owner'],'aggregate_mixed':np.load(ROOT/f'outputs/aggregate_exchange_multipair/layer{l:02d}_mixed_sizes.npz')['owner']}
 for name,tag in [('subset_h1_g8','h1_g8'),('subset_h2_g2','h2_g2'),('subset_h2_g8','h2_g8'),('subset_h2_all','h2_g0')]:expected[name]=np.load(OUT/f'layer{l:02d}_{tag}.npz')['owner']
 times={name:[] for name in methods};orders=[]
 for rep in range(3):
  shift=(l+rep*2)%len(methods);order=methods[shift:]+methods[:shift];orders.append(order)
  for name in order:
   begin=time.perf_counter();dst=run(name,x,seed,su);elapsed=time.perf_counter()-begin
   assert np.array_equal(dst,expected[name]),name
   times[name].append(elapsed)
 rows.append(dict(layer=l,seconds=times,execution_orders=orders));print(l,{k:round(float(np.median(v))*1000,2) for k,v in times.items()},flush=True)
 (OUT/'cpu_benchmark.json').write_text(json.dumps(dict(scope='local CPU, 3 paired rotating-order repetitions per layer; fixed in-memory seed/support; includes candidate mining, selection, filtered index and four rounds; excludes compile, input IO, seed generation, residency selection, physical addresses, GPU communication',python=sys.version,numpy=np.__version__,numba=numba.__version__,platform=platform.platform(),rows=rows),indent=2))
