"""Aggregate full token states, propose once per state, match counts, expand.
CPU semantic prototype. Full source/expert/owner state is a conservative safe
key; no claim of preserving original token-id tie rules or GPU speed.
"""
import time
import numpy as np
from numba import njit

@njit(cache=True)
def proposals(states, support, R, K):
    G=len(states); keys=np.full(G,-1,np.int32)
    for t in range(G):
        src=states[t,0]; occ=np.zeros(R,np.int32)
        for k in range(K):occ[states[t,2+k*2]]+=1
        best=-1; key=-1
        for a in range(R):
            if occ[a]==0:continue
            for b in range(R):
                if a==b:continue
                gain=int(a!=src)-int(b!=src and occ[b]==0)
                if gain<0:continue
                legal=True
                for k in range(K):
                    if states[t,2+k*2]==a and not support[b,states[t,1+k*2]]:
                        legal=False;break
                if not legal:continue
                # Positive gain first, then smaller group, then existing target
                # occupancy; fixed ascending direction tie. No token-id tie.
                score=(gain*(K+1)+(K-occ[a]))*(K+1)+occ[b]
                if score>best:
                    best=score;key=(((a*R+b)*2+gain)*(K+1)+occ[a])
        keys[t]=key
    return keys

@njit(cache=True)
def bounded_dp(counts, cap, K):
    # Maximum gain at each EXACT number of moved compute pairs.
    ntypes=2*K; bad=-100000000
    dp=np.full(cap+1,bad,np.int32);dp[0]=0
    choices=np.zeros((ntypes,cap+1),np.int32)
    queue=np.empty(cap+1,np.int32)
    for i in range(ntypes):
        gain=i//K;size=i%K+1;n=counts[gain,size]
        nxt=np.full(cap+1,bad,np.int32)
        for residue in range(min(size,cap+1)):
            head=0;tail=0
            for j in range((cap-residue)//size+1):
                while head<tail and queue[head]<j-n:head+=1
                pos=residue+j*size
                val=dp[pos]-j*gain
                while head<tail:
                    q=queue[tail-1]
                    if dp[residue+q*size]-q*gain>val:break
                    tail-=1
                queue[tail]=j;tail+=1
                q=queue[head];old=dp[residue+q*size]
                if old>bad//2:
                    nxt[pos]=old+(j-q)*gain
                    choices[i,pos]=j-q
        dp=nxt
    return dp,choices

@njit(cache=True)
def match_counts(counts,R,K,mixed):
    accept=np.zeros_like(counts);pred=0;work=0;maxcap=0
    for a in range(R):
        for b in range(a+1,R):
            if not mixed:
                for s in range(1,K+1):
                    n=min(counts[a,b,0,s]+counts[a,b,1,s],counts[b,a,0,s]+counts[b,a,1,s])
                    n=min(n,max(counts[a,b,1,s],counts[b,a,1,s]))
                    accept[a,b,1,s]=min(n,counts[a,b,1,s]);accept[a,b,0,s]=n-accept[a,b,1,s]
                    accept[b,a,1,s]=min(n,counts[b,a,1,s]);accept[b,a,0,s]=n-accept[b,a,1,s]
                    pred+=accept[a,b,1,s]+accept[b,a,1,s]
            else:
                wa=0;wb=0
                for s in range(1,K+1):
                    wa+=s*(counts[a,b,0,s]+counts[a,b,1,s])
                    wb+=s*(counts[b,a,0,s]+counts[b,a,1,s])
                cap=min(wa,wb)
                if cap==0:continue
                da,ca=bounded_dp(counts[a,b],cap,K);db,cb=bounded_dp(counts[b,a],cap,K)
                work+=4*K*(cap+1);maxcap=max(maxcap,cap)
                best=0;w=0
                for q in range(1,cap+1):
                    if da[q]+db[q]>best:best=da[q]+db[q];w=q
                if best<=0:continue
                pred+=best
                for direction in range(2):
                    remain=w
                    for i in range(2*K-1,-1,-1):
                        size=i%K+1;gain=i//K
                        n=ca[i,remain] if direction==0 else cb[i,remain]
                        if direction==0:accept[a,b,gain,size]=n
                        else:accept[b,a,gain,size]=n
                        remain-=n*size
                    assert remain==0
    return accept,pred,work,maxcap

@njit(cache=True)
def expand(old, inverse, keys, quotas,R,K):
    out=old.copy(); used=np.zeros(len(keys),np.int64);moved=0;tokens=0
    for t in range(len(old)):
        g=inverse[t];key=keys[g]
        if key<0 or used[g]>=quotas[g]:continue
        used[g]+=1;tokens+=1
        edge=key//(K+1)//2;a=edge//R;b=edge%R
        for k in range(K):
            if old[t,k]==a:out[t,k]=b;moved+=1
    assert np.all(used==quotas)
    return out,moved,tokens

def optimize_aggregate(routes,owner,support,rounds=4,mixed=False):
    R,S,K=routes.shape;T=R*S;x=routes.reshape(T,K);dst=owner.reshape(T,K).copy()
    sort=np.argsort(x,axis=1,kind='stable');ex=np.take_along_axis(x,sort,axis=1)
    source=np.repeat(np.arange(R,dtype=np.int32),S);history=[]
    for step in range(rounds):
        start=time.perf_counter();o=np.take_along_axis(dst,sort,axis=1)
        state=np.empty((T,1+2*K),np.int32);state[:,0]=source;state[:,1::2]=ex;state[:,2::2]=o
        unique,inv,mult=np.unique(state,axis=0,return_inverse=True,return_counts=True)
        inv=inv.reshape(-1);group_time=time.perf_counter()-start
        start=time.perf_counter();keys=proposals(unique,support,R,K)
        hist=np.zeros(R*R*2*(K+1),np.int64)
        ok=keys>=0;np.add.at(hist,keys[ok],mult[ok]);hist=hist.reshape(R,R,2,K+1)
        propose_time=time.perf_counter()-start
        start=time.perf_counter();accept,pred,work,maxcap=match_counts(hist,R,K,mixed)
        # Deterministic allocation of accepted count to distinct expert states.
        left=accept.ravel().copy();quota=np.zeros(len(unique),np.int64)
        for g,key in enumerate(keys):
            if key>=0:
                n=min(mult[g],left[key]);quota[g]=n;left[key]-=n
        assert not left.any();match_time=time.perf_counter()-start
        start=time.perf_counter();dst,moved,nt=expand(dst,inv,keys,quota,R,K);expand_time=time.perf_counter()-start
        history.append(dict(round=step+1,unique_states=len(unique),tokens=T,predicted_gain=int(pred),
            moved_pairs=int(moved),accepted_tokens=int(nt),dp_cells=int(work),max_dp_capacity=int(maxcap),
            cpu_seconds=dict(group=group_time,propose=propose_time,match=match_time,expand=expand_time)))
    return dst.reshape(routes.shape),history

@njit(cache=True)
def pair_candidates(states,remaining,support,a,b,K,step):
    keys=np.full(len(states),-1,np.int32)
    for g in range(len(states)):
        if remaining[g]==0:continue
        src=states[g,0];na=0;nb=0;legalab=True;legalba=True
        for k in range(K):
            e=states[g,1+2*k];o=states[g,2+2*k]
            if o==a:
                na+=1
                if not support[b,e]:legalab=False
            if o==b:
                nb+=1
                if not support[a,e]:legalba=False
        va=int(a!=src)-int(b!=src and nb==0)
        vb=int(b!=src)-int(a!=src and na==0)
        okab=na>0 and legalab and va>=0;okba=nb>0 and legalba and vb>=0
        if not okab and not okba:continue
        # At most one direction/state in this card pair; remaining tokens may
        # be offered to other card pairs later in this round.
        direction=0
        if not okab:direction=1
        elif okba:
            sa=va*(K+1)+(K-na);sb=vb*(K+1)+(K-nb)
            if sb>sa or (sb==sa and (g+step+a+b)%2):direction=1
        if direction==0:keys[g]=va*(K+1)+na
        else:keys[g]=(2+vb)*(K+1)+nb
    return keys

@njit(cache=True)
def expand_actions(old,inverse,actions,R,K):
    out=old.copy();cursor=np.zeros(len(actions),np.int32);left=actions.copy();moved=0;nt=0
    for t in range(len(old)):
        g=inverse[t];edge=cursor[g]
        while edge<R*R and left[g,edge]==0:edge+=1
        cursor[g]=edge
        if edge==R*R:continue
        left[g,edge]-=1;a=edge//R;b=edge%R;nt+=1
        for k in range(K):
            if old[t,k]==a:out[t,k]=b;moved+=1
    assert not np.any(left)
    return out,moved,nt

def optimize_aggregate_multipair(routes,owner,support,rounds=4,mixed=True):
    R,S,K=routes.shape;T=R*S;x=routes.reshape(T,K);dst=owner.reshape(T,K).copy()
    sort=np.argsort(x,axis=1,kind='stable');ex=np.take_along_axis(x,sort,axis=1)
    source=np.repeat(np.arange(R,dtype=np.int32),S);history=[]
    pairs=[(a,b) for a in range(R) for b in range(a+1,R)]
    for step in range(rounds):
        start=time.perf_counter();o=np.take_along_axis(dst,sort,axis=1)
        state=np.empty((T,1+2*K),np.int32);state[:,0]=source;state[:,1::2]=ex;state[:,2::2]=o
        unique,inv,mult=np.unique(state,axis=0,return_inverse=True,return_counts=True);inv=inv.reshape(-1)
        group_time=time.perf_counter()-start;start=time.perf_counter()
        remaining=mult.copy();actions=np.zeros((len(unique),R*R),np.int64);pred=0;work=0;maxcap=0
        shift=(step*7)%len(pairs)
        for a,b in pairs[shift:]+pairs[:shift]:
            keys=pair_candidates(unique,remaining,support,a,b,K,step)
            valid=keys>=0;hist=np.zeros(4*(K+1),np.int64);np.add.at(hist,keys[valid],remaining[valid]);hist=hist.reshape(2,2,K+1)
            two=np.zeros((2,2,2,K+1),np.int64);two[0,1]=hist[0];two[1,0]=hist[1]
            ac,gain,cells,cap=match_counts(two,2,K,mixed);pred+=gain;work+=cells;maxcap=max(maxcap,cap)
            for direction in range(2):
                for v in range(2):
                    for size in range(1,K+1):
                        n=ac[direction,1-direction,v,size]
                        if n==0:continue
                        ids=np.flatnonzero(keys==((2*direction+v)*(K+1)+size));counts=remaining[ids]
                        take=np.clip(n-(np.cumsum(counts)-counts),0,counts)
                        assert take.sum()==n
                        actions[ids,(a*R+b) if direction==0 else (b*R+a)]+=take;remaining[ids]-=take
        planning_time=time.perf_counter()-start;start=time.perf_counter()
        dst,moved,nt=expand_actions(dst,inv,actions,R,K)
        history.append(dict(round=step+1,unique_states=len(unique),tokens=T,predicted_gain=int(pred),moved_pairs=int(moved),accepted_tokens=int(nt),dp_cells=int(work),max_dp_capacity=int(maxcap),cpu_seconds=dict(group=group_time,planning=planning_time,expand=time.perf_counter()-start)))
    return dst.reshape(routes.shape),history
