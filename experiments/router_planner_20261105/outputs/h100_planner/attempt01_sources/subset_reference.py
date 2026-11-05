"""Independent CPU semantics for simultaneous subset variant (new schedule)."""
from pathlib import Path
import sys
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from vendor.pruned_subset_exchange import prepare
from vendor.aggregate_exchange import match_counts
from vendor.bucket_optimizer import _edge_tie

@njit(cache=True)
def propose_parallel(x,o,flat,S,E,step,policy):
 T,K=x.shape;counts=np.zeros((8,8,2,3),np.int64);keys=np.full(T,-1,np.int32);masks=np.zeros(T,np.int32);base=E+E*E
 for t in range(T):
  occ=np.zeros(8,np.int32)
  for k in range(K):occ[o[t,k]]+=1
  best=-1
  for val in flat:
   edge=val//base;pat=val%base;a=edge//8;b=edge%8;size=1 if pat<E else 2
   if occ[a]<size:continue
   gain=int(a!=t//S and size==occ[a])-int(b!=t//S and occ[b]==0)
   score=gain*3+3-size
   if policy:score=((gain*2+int((size-1)==((t//28+step)%2)))*9+occ[b])*56+_edge_tie(a,b,t,step,8)
   if gain<0 or score<=best:continue
   e1=pat if pat<E else (pat-E)//E;e2=-1 if pat<E else (pat-E)%E;m=0;sz=0
   for k in range(K):
    if o[t,k]==a and (x[t,k]==e1 or x[t,k]==e2):m|=1<<k;sz+=1
   if sz!=size:continue
   best=score;keys[t]=(a*8+b)*6+gain*3+size;masks[t]=m
   if not policy and best==5:break
  if keys[t]>=0:
   key=keys[t];edge=key//6;gain=(key%6)//3;size=key%3;counts[edge//8,edge%8,gain,size]+=1
 return keys,masks,counts

@njit(cache=True)
def commit_parallel(o,keys,masks,accept):
 result=o.copy()
 for t in range(len(o)):
  key=keys[t]
  if key<0:continue
  edge=key//6;a=edge//8;b=edge%8;gain=(key%6)//3;size=key%3
  if accept[a,b,gain,size]==0:continue
  accept[a,b,gain,size]-=1
  for k in range(o.shape[1]):
   if masks[t]&(1<<k):result[t,k]=b
 assert not np.any(accept)
 return result

def run(routes,owner,support,h=2,g=8,rounds=4,policy=1):
 R,S,K=routes.shape;E=support.shape[1];prepared,diag=prepare(routes,owner,support,h,g,True)
 flat=[]
 for (a,b),rows in prepared[3].items():
  for old,new,pat in rows:flat.append((old*8+new)*(E+E*E)+pat)
 flat=np.array(flat,np.int64);x=routes.reshape(-1,K);o=owner.reshape(-1,K).copy();history=[]
 for step in range(rounds):
  keys,masks,counts=propose_parallel(x,o,flat,S,E,step,policy)
  accept,gain,_,_=match_counts(counts,8,2,True)
  o=commit_parallel(o,keys,masks,accept);history.append(int(gain))
 return o.reshape(routes.shape),history

if __name__=='__main__':
 import json
 from moonep_cpu import communication
 output=ROOT/'outputs/h100_subset_reference';output.mkdir(exist_ok=True);rows=[]
 for l in range(8):
  x=np.load(ROOT/f'outputs/trace_audit/layer_{l:02d}.npz')['topk'].astype(np.int32)
  seed=np.load(ROOT/f'outputs/rt2_b8_comparison/layer{l:02d}_native_rt0.npz')['owner']
  support=np.load(ROOT/f'outputs/replay/layer{l:02d}_b8_small256_global_4.npz')['support']
  dst,hist=run(x,seed,support);before=communication(x,seed,352);after=communication(x,dst,352)
  assert before['pair_loads']==after['pair_loads'] and np.all(support[dst,x])
  assert before['remote_token_destinations']-after['remote_token_destinations']==sum(hist)
  np.savez_compressed(output/f'layer{l:02d}.npz',owner=dst)
  row=dict(layer=l,baseline=before['remote_token_destinations'],metrics=after,round_gains=hist);rows.append(row)
  print(l,after['remote_token_destinations'],100*sum(hist)/before['remote_token_destinations'],flush=True)
 (output/'results.json').write_text(json.dumps({'scope':'CPU reference for new simultaneous token-first subset GPU algorithm','rows':rows},indent=2))
