from pathlib import Path
import json,time,hashlib
import numpy as np
from moonep_cpu import plan,communication,retention_audit
root=Path(__file__).resolve().parents[1];out=root/'outputs/planner_audit';out.mkdir(exist_ok=True)
rows=[]
for f in sorted((root/'outputs/trace_audit').glob('layer_??.npz')):
 x=np.load(f)['topk']
 for rt in (0,1,2,4,7):
  t=time.perf_counter();p=plan(x,E=352,B=44,replica_target=rt);elapsed=time.perf_counter()-t
  row=dict(layer=int(f.stem[-2:]),B=44,replica_target=rt,cpu_model_seconds=elapsed,
           **communication(x,p.owner,E=352),**retention_audit(x,p),
           unsupported_physical_pairs=int((p.physical_topk<0).sum()),
           balance_iterations=p.balance_iterations,allocation_iterations=p.allocation_iterations,
           owner_sha256=hashlib.sha256(p.owner.tobytes()).hexdigest())
  rows.append(row)
  if rt==0:
   np.savez_compressed(out/(f.stem+'_baseline.npz'),owner=p.owner,alloc=p.alloc,tpe=p.tpe,experts_to_copy=p.experts_to_copy)
  print(row,flush=True)
(out/'baseline_replay.json').write_text(json.dumps(rows,indent=2)+'\n')
