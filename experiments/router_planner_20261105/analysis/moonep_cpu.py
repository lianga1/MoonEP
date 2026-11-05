"""NumPy replay of the supplied MoonEP planner's allocation and FIFO mapping.

Source: inputs/MoonEP/tests/planning_reference.py and moonep/planning.py.
This reproduces actual C2 rank/expert-prefix routing, including retention's
lack of a source-priority override. It is a semantic CPU model, not GPU timing.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Plan:
    owner: np.ndarray             # [R,S,K] destination for every execution pair
    alloc: np.ndarray             # [E,R] actual assigned pair counts
    tpe: np.ndarray               # [R,E] source expert counts
    retained: np.ndarray          # [R,E] requested A0 mirrors
    experts_to_copy: np.ndarray   # [R,B], -1 for unused slots
    physical_topk: np.ndarray     # [R,S,K], -1 for unsupported placement
    source_order: np.ndarray      # stable expert-sorted global pair indices
    migration_quotas: np.ndarray  # [R,R] home-group to destination quotas
    balance_iterations: int
    allocation_iterations: list


def plan(topk, E=352, B=None, replica_target=0):
    x=np.asarray(topk)
    if x.ndim!=3 or x.dtype.kind not in 'iu':
        raise ValueError('topk must be integer [R,S,K]')
    R,S,K=x.shape; CAP=S*K
    if E%R or x.min()<0 or x.max()>=E:raise ValueError('invalid expert/rank dimensions')
    epn=E//R
    if B is None:B=epn
    if not 0<=replica_target<=B-1:raise ValueError('replica_target must be between 0 and B-1')
    tpe=np.array([np.bincount(row.ravel(),minlength=E) for row in x],dtype=np.int64)
    counts=tpe.sum(0)
    retained=np.zeros((R,E),dtype=bool)
    for d in range(R):
        candidates=np.flatnonzero((np.arange(E)//epn!=d)&(tpe[d]>0))
        order=np.lexsort((-candidates,-tpe[d,candidates]))
        retained[d,candidates[order[:replica_target]]]=True
    retained_e=(tpe*retained).sum(0)
    retained_d=(tpe*retained).sum(1)
    adj=counts-retained_e
    balance=adj.reshape(R,epn).sum(1)-CAP+retained_d
    alloc=np.zeros((E,R),dtype=np.int64)
    alloc[np.arange(E),np.arange(E)//epn]=adj
    alloc+=(tpe*retained).T
    z=np.zeros((R,R),dtype=np.int64)
    n_balance=0
    while balance.max()>0:
        h=int(balance.argmax()); d=int(balance.argmin())
        if balance[d]>=0:raise AssertionError('no deficit')
        take=-balance[d]
        z[h,d]=take; balance[h]-=take;balance[d]=0
        n_balance+=1
        if n_balance>R:raise AssertionError('balance did not terminate')
    n_alloc=[]
    for h in range(R):
        rem=adj[h*epn:(h+1)*epn].copy();quota=z[h].copy();steps=0
        while quota.max()>0:
            d=int(quota.argmax());local_e=int(rem.argmax());e=h*epn+local_e
            take=min(rem[local_e],quota[d])
            if take<=0:raise AssertionError('exhausted home group')
            alloc[e,d]+=take;alloc[e,h]-=take;rem[local_e]-=take;quota[d]-=take;steps+=1
        n_alloc.append(steps)
    if (alloc<0).any() or not np.array_equal(alloc.sum(1),counts):raise AssertionError('expert conservation')
    if not np.array_equal(alloc.sum(0),np.full(R,CAP)):raise AssertionError('rank balance')
    etc=np.full((R,B),-1,dtype=np.int32)
    slot_lut=np.full((R,E),-1,dtype=np.int32)
    for d in range(R):
        slot_lut[d,d*epn:(d+1)*epn]=np.arange(epn,dtype=np.int32)
        remote=np.flatnonzero((np.arange(E)//epn!=d)&(alloc[:,d]>0))
        order=np.lexsort((-remote,-alloc[remote,d],-retained[d,remote].astype(np.int32)))
        chosen=remote[order[:B]]
        etc[d,:len(chosen)]=chosen
        slot_lut[d,chosen]=epn+np.arange(len(chosen))
    # Stable sort is exactly the C1 source order: source-rank then flat top-k.
    flat=x.ravel(); order=np.argsort(flat,kind='stable')
    sorted_experts=flat[order]
    starts=np.repeat(np.cumsum(counts)-counts,counts)
    global_occurrence=np.arange(flat.size,dtype=np.int64)-starts
    prefix=alloc.cumsum(1)
    sorted_dest=(global_occurrence[:,None]>=prefix[sorted_experts]).sum(1)
    owner=np.empty(flat.size,dtype=np.int16);owner[order]=sorted_dest
    local_slot=slot_lut[owner,flat]
    physical=np.where(local_slot>=0,owner.astype(np.int32)*(epn+B)+local_slot,-1)
    return Plan(owner.reshape(x.shape),alloc,tpe,retained,etc,physical.reshape(x.shape),order,z,n_balance,n_alloc)


def communication(topk, owner, E=None):
    x=np.asarray(topk);o=np.asarray(owner)
    R,S,K=x.shape
    if E is None:E=int(x.max())+1
    if o.shape!=x.shape or (o<0).any() or (o>=R).any():raise ValueError('invalid assignment')
    # [source,token,destination] keeps exactly one payload per remote destination.
    visited=(o[...,None]==np.arange(R)).any(axis=2)
    remote=visited.copy()
    for r in range(R):remote[r,:,r]=False
    per_dest=remote.sum(axis=(0,1));per_source=remote.sum(axis=(1,2))
    loads=np.bincount(o.ravel(),minlength=R)
    exec_counts=np.bincount(o.ravel().astype(np.int64)*E+x.ravel(),minlength=R*E).reshape(R,E)
    home=np.arange(E)//(E//R)
    active_extra=((exec_counts>0)&(home[None,:]!=np.arange(R)[:,None])).sum(1)
    return dict(remote_token_destinations=int(remote.sum()),all_token_destinations=int(visited.sum()),
                remote_pairs=int((o!=np.arange(R)[:,None,None]).sum()),
                source_remote_payloads=per_source.tolist(),destination_remote_payloads=per_dest.tolist(),
                pair_loads=loads.tolist(),max_min_pair_load=int(loads.max()-loads.min()),
                max_mean_pair_load=float(loads.max()/loads.mean()),active_extra_experts=active_extra.tolist())


def retention_audit(topk, p):
    x=np.asarray(topk);R=x.shape[0]
    intended=p.retained[np.arange(R)[:,None,None],x]
    local=p.owner==np.arange(R)[:,None,None]
    n=int(intended.sum())
    return dict(intended_retained_pairs=n,actually_local_intended_pairs=int((intended&local).sum()),
                intended_pairs_routed_remote=int((intended&~local).sum()),
                retained_local_fraction=float((intended&local).sum()/n) if n else None)
