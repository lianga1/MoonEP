"""Independent CPU oracle for native MoonEP metadata from explicit task owners.

Not used by timed execution. Canonical address order is global source/token/
expert-choice order within each destination/physical expert segment.
"""
import numpy as np


def reference_native_plan(routes, owner, support, *, B=8, padding=128, NvS=None):
    routes=np.asarray(routes);owner=np.asarray(owner);support=np.asarray(support)
    R,S,K=routes.shape;E=support.shape[1];epn=E//R;Q=E+B;M=S*K
    assert owner.shape==routes.shape and support.shape==(R,E) and E%R==0
    assert np.all((owner>=0)&(owner<R)) and np.all((routes>=0)&(routes<E))
    NvS=M+(padding-1)*2*epn if NvS is None else NvS
    ids=np.full((R,B),-1,np.int32);lut=np.full((R,E),-1,np.int32)
    for r in range(R):
        lut[r,r*epn:(r+1)*epn]=np.arange(r*epn,(r+1)*epn)
        extra=np.flatnonzero(support[r]&(np.arange(E)//epn!=r))
        assert len(extra)<=B
        ids[r,:len(extra)]=extra
        lut[r,extra]=E+np.arange(len(extra))
    r=owner.ravel();e=routes.ravel();segment=lut[r,e]
    assert np.all(segment>=0)
    key=r*Q+segment
    counts=np.bincount(key,minlength=R*Q).reshape(R,Q)
    assert np.all(counts.sum(1)==M)
    padded=((counts+padding-1)//padding)*padding
    cu=np.cumsum(padded,axis=1,dtype=np.int32)
    starts=cu-padded
    assert np.max(cu[:,-1])<=NvS
    zfr=np.zeros((R,Q,2),np.int32)
    haspad=padded>counts
    zfr[:,:,0]=np.where(haspad,starts+counts,0)
    zfr[:,:,1]=padded-counts
    physical=r*(epn+B)+np.where(segment<E,e%epn,epn+segment-E)
    raw=np.empty(R*M,np.int32);src=np.full((R,NvS),-1,np.int32)
    order=np.argsort(key,kind='stable')
    ordinal=np.zeros(R*Q,np.int32)
    for task in order:
        dest=int(r[task]);group=int(segment[task]);slot=int(starts[dest,group]+ordinal[key[task]])
        ordinal[key[task]]+=1
        raw[task]=dest*NvS+slot
        src[dest,slot]=(task//M)*NvS+task%M
    dst=raw.reshape(R*S,K).copy()
    for t in range(R*S):
        seen=set()
        for k in range(K):
            dest=int(owner.reshape(-1,K)[t,k])
            if dest in seen:dst[t,k]=-dst[t,k]-1
            seen.add(dest)
    remote_stats=np.zeros((R,2),np.int32)
    for dest in range(R):
        for b in range(B):
            if counts[dest,E+b]>0:
                remote_stats[dest,0]+=1
                remote_stats[int(ids[dest,b])//epn,1]+=1
    return dict(dst=dst.reshape(R,S,K),raw_dst=raw.reshape(R,S,K),physical_topk=physical.astype(np.int32).reshape(R,S,K),
                experts_to_copy=ids,cu_seqlens=cu,zero_fill_ranges=zfr,src_info=src,counts=counts,remote_stats=remote_stats)
