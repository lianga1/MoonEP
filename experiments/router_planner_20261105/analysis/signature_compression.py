"""Exact signature counts on original tokens and native seed, not speed estimates."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for layer in range(8):
 x=np.load(ROOT/f'outputs/trace_audit/layer_{layer:02d}.npz')['topk']
 owner=np.load(ROOT/f'outputs/planner_audit/layer_{layer:02d}_baseline.npz')['owner']
 R,S,K=x.shape
 perrank=[]
 for rank in range(R):
  order=np.argsort(x[rank],axis=1,kind='stable')
  experts=np.take_along_axis(x[rank],order,axis=1)
  dests=np.take_along_axis(owner[rank],order,axis=1)
  state=np.stack([experts,dests],axis=-1).reshape(S,2*K)
  _,route_counts=np.unique(experts,axis=0,return_counts=True)
  _,state_counts=np.unique(state,axis=0,return_counts=True)
  perrank.append(dict(source=rank,tokens=S,route_unique=len(route_counts),seed_state_unique=len(state_counts),
                      route_singleton_signatures=int((route_counts==1).sum()),seed_state_singleton_signatures=int((state_counts==1).sum()),
                      route_max_multiplicity=int(route_counts.max()),seed_state_max_multiplicity=int(state_counts.max())))
 T=R*S;Ur=sum(r['route_unique'] for r in perrank);Us=sum(r['seed_state_unique'] for r in perrank)
 rows.append(dict(layer=layer,tokens=T,route_unique_source_expertset=Ur,seed_state_unique_source_sorted_expert_owner=Us,
                  route_duplicate_fraction=1-Ur/T,seed_state_duplicate_fraction=1-Us/T,
                  route_token_to_unique_ratio=T/Ur,seed_state_token_to_unique_ratio=T/Us,
                  state_unique_increase_vs_route_percent=100*(Us-Ur)/Ur,per_source=perrank))
 print(layer,T,Ur,Us,round(100*(1-Ur/T),3),round(100*(1-Us/T),3),flush=True)
tokens=sum(r['tokens'] for r in rows);Ur=sum(r['route_unique_source_expertset'] for r in rows);Us=sum(r['seed_state_unique_source_sorted_expert_owner'] for r in rows)
out=dict(scope='Exact per-layer signature compression counts; not measured planner work reduction or speedup',
         route_key='(layer,source_rank,sorted expert IDs)',seed_state_key='(layer,source_rank,sorted (expert_id,seed_owner))',
         total_tokens_over_layers=tokens,total_route_unique=Ur,total_seed_state_unique=Us,
         total_route_duplicate_fraction=1-Ur/tokens,total_seed_state_duplicate_fraction=1-Us/tokens,
         warnings=['Token-id-dependent edge/size tie-breaking is not in either signature.',
                   'Per-expert occurrence and physical execution row addresses remain token-specific.',
                   'Identical seed state may diverge after quota acceptance; signatures must be updated for subsequent rounds.',
                   'Expert weights and residency differ by layer; keys are never merged across layers.',
                   'These counts do not prove the GPU hash/group/scatter overhead is repaid.'],layers=rows)
p=ROOT/'outputs/planner_audit/signature_compression.json';p.write_text(json.dumps(out,indent=2)+'\n')
print('aggregate',tokens,Ur,Us,1-Ur/tokens,1-Us/tokens)
