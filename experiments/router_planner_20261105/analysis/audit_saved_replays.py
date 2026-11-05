"""Independent audit of saved owner/support mappings, without optimizer imports."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1];rows=json.loads((root/'outputs/replay/strategies.json').read_text())['rows']
checks=[]
for row in rows:
 l=row['layer'];B=row['budget'];x=np.load(root/f'outputs/trace_audit/layer_{l:02d}.npz')['topk'];R,S,K=x.shape
 seed=np.load(root/f'outputs/planner_audit/layer_{l:02d}_baseline.npz')['owner']
 for v in row['variants']:
  a=np.load(root/f'outputs/replay/layer{l:02d}_b{B}_{v["name"]}.npz');o=a['owner'];support=a['support']
  assert o.shape==x.shape and np.all((o>=0)&(o<R))
  assert np.all(support[o,x])
  local_loads=np.array([np.bincount(o[r].ravel(),minlength=R) for r in range(R)])
  seed_local=np.array([np.bincount(seed[r].ravel(),minlength=R) for r in range(R)])
  assert np.array_equal(local_loads.sum(0),seed_local.sum(0))
  if 'source_sharded' in v['name']:assert np.array_equal(local_loads,seed_local)
  home=np.arange(352)//44
  assert np.max((support&(np.arange(R)[:,None]!=home)).sum(1))<=B
  ordered=np.sort(o,axis=-1)
  first=np.concatenate([np.ones((R,S,1),bool),ordered[:,:,1:]!=ordered[:,:,:-1]],axis=-1)
  remote=int((first&(ordered!=np.arange(R)[:,None,None])).sum())
  assert remote==v['metrics']['remote_token_destinations']
  assert remote<=row['baseline']['remote_token_destinations']
  checks.append(dict(layer=l,budget=B,variant=v['name'],remote_token_destinations=remote,pass_all=True,
                     per_source_execution_loads_preserved=bool(np.array_equal(local_loads,seed_local))))
out=dict(scope='CPU saved owner/support semantic audit; no physical output address or GPU validation',count=len(checks),checks=checks)
(root/'outputs/planner_audit/replay_independent_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('independent audited',len(checks),'plans')
