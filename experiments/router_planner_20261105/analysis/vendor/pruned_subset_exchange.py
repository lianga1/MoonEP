"""Static expert-subset shortlist + reusable token posting index, CPU prototype.
No full-state grouping. Shortlist built once from initial owner, exact full trace.
Each round inspects only shortlisted pattern postings, with exact dynamic gain
and one-token-per-round consumption. h=1/2; g=0 retains all patterns.
"""
import time
import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
from .aggregate_exchange import match_counts

@njit(cache=True)
def mine(x,o,support,R,S,E,h):
    counts=Dict.empty(types.int64,types.int64);K=x.shape[1];base=E+E*E
    examined=0
    for t in range(len(x)):
        src=t//S;occ=np.zeros(R,np.int32)
        for k in range(K):occ[o[t,k]]+=1
        for i in range(K):
            a=o[t,i];e=x[t,i]
            for j in range(-1,K):
                if j>=0 and (h<2 or j<=i or o[t,j]!=a):continue
                size=1 if j<0 else 2
                pattern=e if j<0 else E+min(e,x[t,j])*E+max(e,x[t,j])
                for b in range(R):
                    if a==b or not support[b,e]:continue
                    if j>=0 and not support[b,x[t,j]]:continue
                    examined+=1
                    gain=int(a!=src and size==occ[a])-int(b!=src and occ[b]==0)
                    if gain<0:continue
                    key=((((a*R+b)*2+gain)*2+size-1)*base)+pattern
                    counts[key]=counts.get(key,0)+1
    keys=np.empty(len(counts),np.int64);vals=np.empty(len(counts),np.int64)
    n=0
    for k,v in counts.items():keys[n]=k;vals[n]=v;n+=1
    return keys,vals,examined

@njit(cache=True)
def index_records(x,E,h):
    T,K=x.shape;n=T*(K+(K*(K-1)//2 if h==2 else 0))
    pats=np.empty(n,np.int32);tokens=np.empty(n,np.int32);n=0
    for t in range(T):
        for i in range(K):
            pats[n]=x[t,i];tokens[n]=t;n+=1
            if h==2:
                for j in range(i+1,K):
                    pats[n]=E+min(x[t,i],x[t,j])*E+max(x[t,i],x[t,j]);tokens[n]=t;n+=1
    return pats,tokens

@njit(cache=True)
def collect(pair_patterns,postings,starts,x,o,used,a,b,S,E,K,bestkey,bestmask,touched):
    n=0;scans=0
    # bestkey holds size/gain/direction; touched avoids clearing full T per pair.
    for row in range(len(pair_patterns)):
        old=pair_patterns[row,0];new=pair_patterns[row,1];pat=pair_patterns[row,2]
        e1=pat if pat<E else (pat-E)//E;e2=-1 if pat<E else (pat-E)%E
        size=1 if e2<0 else 2
        direction=0 if old==a else 1
        for ix in range(starts[pat],starts[pat+1]):
            t=postings[ix];scans+=1
            if used[t]:continue
            mask=0;oldcount=0;newcount=0
            for k in range(K):
                if o[t,k]==old:oldcount+=1
                if o[t,k]==new:newcount+=1
                if (x[t,k]==e1 or x[t,k]==e2) and o[t,k]==old:mask|=1<<k
            if int(mask!=0)==0:continue
            # Ensure every pattern expert is still at the required old card.
            sz=0
            for k in range(K):
                if mask&(1<<k):sz+=1
            if sz!=size:continue
            src=t//S;gain=int(old!=src and oldcount==size)-int(new!=src and newcount==0)
            if gain<0:continue
            key=(direction*2+gain)*3+size
            oldkey=bestkey[t]
            score=gain*3+(3-size)
            oldscore=-1 if oldkey<0 else ((oldkey//3)%2)*3+(3-oldkey%3)
            if score>oldscore:
                if oldkey<0:touched[n]=t;n+=1
                bestkey[t]=key;bestmask[t]=mask
    return n,scans

@njit(cache=True)
def commit(touched,n,bestkey,bestmask,accepted,used,o,a,b,K):
    gain=0;moved=0;nt=0
    # Positive and neutral categories have independent exact quotas.
    for i in range(n):
        t=touched[i];key=bestkey[t];direction=key//6;v=(key//3)%2;size=key%3
        if accepted[direction,1-direction,v,size]>0:
            accepted[direction,1-direction,v,size]-=1;used[t]=True;nt+=1;gain+=v
            new=b if direction==0 else a
            for k in range(K):
                if bestmask[t]&(1<<k):o[t,k]=new;moved+=1
        bestkey[t]=-1;bestmask[t]=0
    assert not np.any(accepted)
    return gain,moved,nt

@njit(cache=True)
def histogram(touched,n,bestkey):
    c=np.zeros((2,2,2,3),np.int64)
    for i in range(n):
        key=bestkey[touched[i]];d=key//6;v=(key//3)%2;s=key%3
        c[d,1-d,v,s]+=1
    return c

def prepare(routes,owner,support,h=2,g=8,filtered=False):
    R,S,K=routes.shape;E=support.shape[1];x=np.ascontiguousarray(routes.reshape(-1,K));o=np.ascontiguousarray(owner.reshape(-1,K))
    begin=time.perf_counter();keys,vals,examined=mine(x,o,support,R,S,E,h);mine_seconds=time.perf_counter()-begin
    begin=time.perf_counter();base=E+E*E;by={}
    for key,val in zip(keys,vals):by.setdefault(int(key//base),[]).append((int(val),int(key%base)))
    chosen=set()
    for cls,items in by.items():
        items.sort(key=lambda z:(-z[0],z[1]));edge=cls//4;a=edge//R;b=edge%R
        for _,pat in (items[:g] if g else items):chosen.add((a,b,pat))
    pairs={}
    for a in range(R):
        for b in range(a+1,R):pairs[(a,b)]=np.array(sorted((u,v,p) for u,v,p in chosen if (u==a and v==b) or (u==b and v==a)),np.int64).reshape(-1,3)
    select_seconds=time.perf_counter()-begin
    begin=time.perf_counter()
    if filtered:
        allowed=np.zeros(base,np.bool_)
        for a,b,pat in chosen:allowed[pat]=True
        postings,starts=selected_index(x,E,h,allowed)
    else:
        patterns,tokens=index_records(x,E,h);order=np.argsort(patterns,kind='stable');postings=tokens[order]
        counts=np.bincount(patterns,minlength=base);starts=np.concatenate(([0],np.cumsum(counts)))
    index_seconds=time.perf_counter()-begin
    return (x,postings,starts,pairs),dict(h=h,g=g,raw_classes=len(keys),selected_direction_patterns=len(chosen),mining_candidate_checks=int(examined),posting_entries=len(postings),cpu_prepare_seconds=dict(mine=mine_seconds,select=select_seconds,index=index_seconds))

def execute(routes,owner,support,prepared,rounds=4):
    R,S,K=routes.shape;E=support.shape[1];x,postings,starts,pairs=prepared;T=len(x)
    dst=np.ascontiguousarray(owner.reshape(T,K)).copy();bestkey=np.full(T,-1,np.int32);bestmask=np.zeros(T,np.int32);touched=np.empty(T,np.int32);history=[]
    pp=list(pairs)
    for step in range(rounds):
        used=np.zeros(T,np.bool_);pred=0;scans=0;moved=0;nt=0;dp=0;maxcap=0;begin=time.perf_counter()
        shift=(step*7)%len(pp)
        for a,b in pp[shift:]+pp[:shift]:
            patterns=pairs[(a,b)]
            if len(patterns)==0:continue
            n,s=collect(patterns,postings,starts,x,dst,used,a,b,S,E,K,bestkey,bestmask,touched);scans+=s
            c=histogram(touched,n,bestkey);ac,gain,cells,cap=match_counts(c,2,2,True)
            actual,mv,nv=commit(touched,n,bestkey,bestmask,ac,used,dst,a,b,K)
            assert gain==actual;pred+=gain;moved+=mv;nt+=nv;dp+=cells;maxcap=max(maxcap,cap)
        history.append(dict(round=step+1,predicted_gain=int(pred),posting_visits=int(scans),moved_pairs=int(moved),accepted_tokens=int(nt),dp_cells=int(dp),max_dp_capacity=int(maxcap),cpu_seconds=time.perf_counter()-begin))
    return dst.reshape(routes.shape),history

@njit(cache=True)
def selected_index(x,E,h,allowed):
    # Two linear passes, counting then filling only shortlisted expert patterns.
    # No global sorting and no storage for rejected pattern memberships.
    base=len(allowed);counts=np.zeros(base,np.int64);T,K=x.shape
    for t in range(T):
        for i in range(K):
            e=x[t,i]
            if allowed[e]:counts[e]+=1
            if h==2:
                for j in range(i+1,K):
                    p=E+min(e,x[t,j])*E+max(e,x[t,j])
                    if allowed[p]:counts[p]+=1
    starts=np.empty(base+1,np.int64);starts[0]=0
    for p in range(base):starts[p+1]=starts[p]+counts[p]
    cursor=starts[:-1].copy();postings=np.empty(starts[-1],np.int32)
    for t in range(T):
        for i in range(K):
            e=x[t,i]
            if allowed[e]:postings[cursor[e]]=t;cursor[e]+=1
            if h==2:
                for j in range(i+1,K):
                    p=E+min(e,x[t,j])*E+max(e,x[t,j])
                    if allowed[p]:postings[cursor[p]]=t;cursor[p]+=1
    return postings,starts
