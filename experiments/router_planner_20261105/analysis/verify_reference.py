"""Run in any environment with Torch to compare original allocation reference.
Full C2 destination mapping additionally uses an independent scalar FIFO loop.
"""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
import torch
from moonep_cpu import plan
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supplied_planning_reference',root/'inputs/MoonEP/tests/planning_reference.py')
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
rng=np.random.default_rng(20261026);checks=[]
cases=[]
for R,E,S,K in [(2,8,15,3),(4,16,19,4),(8,352,31,8)]:
 x=np.array([[rng.choice(E,K,replace=False) for _ in range(S)] for r in range(R)],dtype=np.int32)
 cases.append((f'random_R{R}',x,E))
for f in sorted((root/'outputs/trace_audit').glob('layer_??.npz')):cases.append((f.stem,np.load(f)['topk'],352))
for name,x,E in cases:
 R,S,K=x.shape;B=E//R
 for rt in (0,min(2,B-1)):
  p=plan(x,E=E,B=B,replica_target=rt)
  ctx=dict(rank=0,R=R,E=E,B=B,S=S,K=K,NvS_capacity=S*K,replica_target=rt)
  got=ref.launch_planning_torch_reference(ctx,torch.from_numpy(x),torch.from_numpy(p.tpe),return_alloc=True).numpy().T
  np.testing.assert_array_equal(p.alloc,got)
  # Vectorized per-expert FIFO independent of globally sorted occurrence index.
  scalar=np.empty(x.size,dtype=np.int16)
  for e in range(E):
   ix=np.flatnonzero(x.ravel()==e)
   scalar[ix]=np.repeat(np.arange(R),got[e])
  np.testing.assert_array_equal(p.owner.ravel(),scalar)
  checks.append(dict(case=name,shape=list(x.shape),replica_target=rt,alloc_exact=True,destination_exact=True))
print(json.dumps(dict(torch_version=torch.__version__,checks=checks),indent=2))
