"""Greedy residency for heterogeneous hot cores, independent of core home layout.

No solver. Preserve every expert required by the balanced seed; at most B extra
experts per rank including those mandatory copies. Sampled reciprocal bundles
are *estimates* of useful exchanges, never execution instructions.
"""
from collections import Counter, defaultdict
import numpy as np
from numba import njit


@njit(cache=True)
def _votes(routes, owner, draft):
    R,S,K=routes.shape
    T=R*S
    E=352
    out=np.zeros((R,E),np.int64)
    xx=routes.reshape(T,K);dd=owner.reshape(T,K)
    for t in range(T):
        r=t//S
        if draft:
            cnt=np.zeros(R,np.int32)
            for k in range(K):cnt[dd[t,k]]+=1
            a=t%R
            for j in range(1,R):
                q=(t+j)%R
                if cnt[q]>cnt[a]:a=q
            if cnt[r]==cnt[a]:a=r
            r=a
        for k in range(K):out[r,xx[t,k]]+=1
    return out


@njit(cache=True)
def _visited_votes(routes, owner):
    """Earlier residency rule: a token votes at every seed destination it visits."""
    R,S,K=routes.shape;T=R*S;E=352
    out=np.zeros((R,E),np.int64)
    xx=routes.reshape(T,K);dd=owner.reshape(T,K)
    for t in range(T):
        seen=np.zeros(R,np.bool_)
        for k in range(K):seen[dd[t,k]]=True
        for r in range(R):
            if seen[r]:
                for k in range(K):out[r,xx[t,k]]+=1
    return out


def _mandatory(routes, owner, home, slots):
    R=routes.shape[0];E=len(home)
    counts=np.bincount(owner.ravel().astype(np.int64)*E+routes.ravel(),minlength=R*E).reshape(R,E)
    support=(np.arange(R)[:,None]==home[None,:])|(counts>0)
    extra=np.count_nonzero(support & (np.arange(R)[:,None]!=home[None,:]),axis=1)
    if np.max(extra)>slots:
        raise ValueError('balanced MoonEP seed exceeds the hard extra-expert budget')
    return support,extra


def select_support(routes, owner, home, slots=8, method='source', sample_tokens=4096,
                   groups_per_direction=4, seed=20261015):
    """Return support[R,E] and JSON diagnostics; input assignment is unchanged.

    source/draft/visited: fill residual budget by individual-expert frequency.
    reciprocal: sample whole (token, old-rank) bundles; retain bounded top groups
    per direction/size/gain. Greedily buy TWO mutually executable expert groups
    with equal bundle size, using estimated matched savings per additional slot.
    Remaining unmatched slots are left empty, rather than silently filled with
    unrelated high-frequency experts. No assumption about which home owns core.
    """
    R,S,K=routes.shape;E=len(home)
    support,extra=_mandatory(routes,owner,home,slots)
    mandatory=extra.copy()
    if method=='seed':
        return support,dict(method=method,mandatory_extra=mandatory.tolist(),candidate_extra=extra.tolist())
    if method in ('source','draft','visited'):
        freq=_visited_votes(routes,owner) if method=='visited' else _votes(routes,owner,method=='draft')
        for r in range(R):
            missing=np.flatnonzero(~support[r])
            priority=np.lexsort((missing,-freq[r,missing]))
            support[r,missing[priority[:slots-extra[r]]]]=True
        return support,dict(method=method,mandatory_extra=mandatory.tolist(),
                            candidate_extra=np.count_nonzero(support & (np.arange(R)[:,None]!=home[None,:]),axis=1).tolist())
    if method!='reciprocal':raise ValueError(method)
    rng=np.random.default_rng(seed)
    chosen=np.sort(rng.choice(R*S,min(sample_tokens,R*S),replace=False))
    flat=routes.reshape(-1,K);own=owner.reshape(-1,K)
    hist=defaultdict(Counter)
    infeasible_directions=0
    for t in chosen:
        src=int(t//S)
        groups=defaultdict(list)
        for e,a in zip(flat[t],own[t]):groups[int(a)].append(int(e))
        for a in range(R):
            for b in range(a+1,R):
                choices=[]
                for old,new in ((a,b),(b,a)):
                    if old not in groups:continue
                    gain=int(old!=src)-int(new!=src and new not in groups)
                    if gain<0:continue
                    signature=tuple(sorted(groups[old]))
                    # This group can never fit: later copies consume the same
                    # residual slots that they remove from its missing set.
                    # Filter before tie-breaking/Top-g to avoid crowding out
                    # less frequent but feasible groups.
                    if sum(not support[new,e] for e in signature)>slots-extra[new]:
                        infeasible_directions+=1
                        continue
                    choices.append((gain,old,new,signature))
                if not choices:continue
                # One sample token cannot be counted as its own reverse partner.
                # Prefer higher gain, rotate exact ties by a stable token/pair bit.
                maxgain=max(x[0] for x in choices)
                choices=[x for x in choices if x[0]==maxgain]
                # Mix high/low token bits as well as the rank pair. A plain
                # odd multiple modulo two would reduce to token parity.
                tie=(int(t)+a*0x9e3779b9+b*0x85ebca6b)&0xffffffff
                tie^=tie>>16
                tie=(tie*0x7feb352d)&0xffffffff
                tie^=tie>>15
                gain,old,new,signature=choices[tie%len(choices)]
                hist[(old,new,len(signature),gain)][signature]+=1
    # A signature is a collision-free tuple in this CPU prototype. GPU hash
    # implementation must compare exact expert masks on hash collisions.
    tops=defaultdict(list)
    for (a,b,size,gain),counter in hist.items():
        for group,n in sorted(counter.items(),key=lambda x:(-x[1],x[0]))[:groups_per_direction]:
            tops[(a,b,size)].append((group,int(n),gain))
    packages=[]
    for a in range(R):
        for b in range(a+1,R):
            for size in range(1,K+1):
                for ga,na,va in tops.get((a,b,size),[]):
                    for gb,nb,vb in tops.get((b,a,size),[]):
                        if va+vb==0:continue
                        packages.append((a,b,ga,gb,na,nb,va,vb,size))
    # Index sparse potential bundle entries once, to track gains already enabled.
    records={}
    for key,groups in tops.items():
        records[key]=groups
    history=[]
    while True:
        enabled={}
        for key,groups in records.items():
            a,b,size=key
            n0=n1=0
            for g,n,v in groups:
                if all(support[b,e] for e in g):
                    if v:n1+=n
                    else:n0+=n
            enabled[key]=(n0,n1)
        best=None
        for i,(a,b,ga,gb,na,nb,va,vb,size) in enumerate(packages):
            add_b=tuple(e for e in ga if not support[b,e])
            add_a=tuple(e for e in gb if not support[a,e])
            cost=len(add_a)+len(add_b)
            if not cost or extra[a]+len(add_a)>slots or extra[b]+len(add_b)>slots:continue
            x0,x1=enabled[(a,b,size)];y0,y1=enabled[(b,a,size)]
            oldn=min(x0+x1,y0+y1,max(x1,y1))
            oldgain=min(oldn,x1)+min(oldn,y1)
            # A new group can complete several previously blocked groups.
            # Recount all retained groups in this direction/size, not just ga/gb.
            x0=x1=y0=y1=0
            for group,n,value in records[(a,b,size)]:
                if all(support[b,e] or e in add_b for e in group):
                    x0+=n*(1-value);x1+=n*value
            for group,n,value in records[(b,a,size)]:
                if all(support[a,e] or e in add_a for e in group):
                    y0+=n*(1-value);y1+=n*value
            newn=min(x0+x1,y0+y1,max(x1,y1))
            gain=min(newn,x1)+min(newn,y1)-oldgain
            if gain<=0:continue
            score=(gain/cost,gain,-cost,-i)
            if best is None or score>best[0]:best=(score,a,b,add_a,add_b,size,gain)
        if best is None:break
        _,a,b,add_a,add_b,size,gain=best
        support[a,list(add_a)]=True; support[b,list(add_b)]=True
        extra[a]+=len(add_a);extra[b]+=len(add_b)
        history.append(dict(a=a,b=b,add_a=list(add_a),add_b=list(add_b),size=size,estimated_sample_gain=int(gain)))
        assert len(history)<=R*slots
    assert np.max(extra)<=slots
    return support,dict(method=method,mandatory_extra=mandatory.tolist(),candidate_extra=extra.tolist(),
                        sampled_tokens=len(chosen),distinct_group_buckets=len(hist),
                        infeasible_directions_filtered=infeasible_directions,
                        candidate_packages=len(packages),accepted_packages=history,
                        warning='one direction per sample token/rank-pair; cross-pair overlaps remain estimates; final exchanges audited independently')
