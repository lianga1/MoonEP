"""Create labelled synthetic correctness fixtures, never performance evidence."""
from pathlib import Path
import hashlib
import json
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
try:
    import numba
except ImportError:
    stub=types.ModuleType('numba');stub.njit=lambda *a,**kw:a[0] if a and callable(a[0]) else lambda f:f
    sys.modules['numba']=stub
sys.path.insert(0,str(ROOT/'analysis/vendor'))
from residency import select_support
from residency_reference_check import emulate


def save_case(out,name,x,owner,d,seed):
    S=x.shape[1]
    etc=np.full((2,8),-1,np.int32)
    for rank in range(2):
        experts=np.unique(x[(owner==rank)&(x//176!=rank)])
        assert len(experts)<=8
        etc[rank,:len(experts)]=experts
    support,diag=select_support(x,owner,np.arange(352)//176,slots=8,method='reciprocal',
        sample_tokens=d,groups_per_direction=2,seed=seed)
    ids=np.sort(np.random.default_rng(seed).choice(2*S,min(d,2*S),replace=False))
    sample=x.reshape(-1,8)[ids]
    history=[[h['a'],h['b'],len(h['add_a']),len(h['add_b']),h['size'],h['estimated_sample_gain'],len(h['add_a'])+len(h['add_b'])]
             for h in diag['accepted_packages']]
    actual,ah,ad,neutral=emulate(sample,ids,owner.reshape(-1,8)[ids],etc,S)
    assert np.array_equal(actual,support) and ah==history and ad==[diag['candidate_packages'],*diag['candidate_extra']]
    path=out/(name+'.npz')
    np.savez_compressed(path,topk=x,owner=owner,native_experts_to_copy=etc,support=support,
        sample_ids=ids.astype(np.int64),sample_routes=sample)
    return dict(name=name,file=path.name,shape=list(x.shape),samples=len(ids),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),history=history,
        diagnostics=[diag['candidate_packages'],*diag['candidate_extra']],gain_zero_samples=neutral)


def main():
    out=ROOT/'outputs/two_gpu_overlap/residency_corners';out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for count in (3,5,7,8):
        for seed in (19,23,41):
            rng=np.random.default_rng(seed);S=256
            x=np.stack([rng.choice(352,8,replace=False) for _ in range(2*S)]).astype(np.int32).reshape(2,S,8)
            support=np.arange(352)[None,:]//176==np.arange(2)[:,None]
            support[1,rng.choice(176,count,replace=False)]=True
            owner=x//176
            for r in range(2):
                for t in range(S):
                    for k,e in enumerate(x[r,t]):
                        if support[0,e] and support[1,e]:owner[r,t,k]=rng.integers(0,2)
            cases.append(save_case(out,f'budget{count}_seed{seed}',x,owner,256,seed))
    # Each class has gain-0 and gain-1 groups in both directions. This creates
    # P=80 packages, exercises warp-strided candidates, and requires preserving
    # category insertion order. Mandatory copies are forced by the final row.
    rank_rows=[[],[]];rank_owner=[[],[]]
    for rank in range(2):
        remote_base=(1-rank)*176;local_base=rank*176
        for size in range(1,9):
            for choice in range(2):
                group=list(range(remote_base,remote_base+size-1))+[remote_base+7+choice]
                rest=list(range(local_base+20,local_base+20+8-size))
                rank_rows[rank].append(group+rest);rank_owner[rank].append([1-rank]*size+[rank]*(8-size))
        for size in range(1,7):
            for choice in range(2):
                group=list(range(local_base,local_base+size-1))+[local_base+7+choice]
                rest=list(range(remote_base+30,remote_base+30+8-size))
                rank_rows[rank].append(group+rest);rank_owner[rank].append([rank]*size+[1-rank]*(8-size))
        rank_rows[rank].append(list(range(remote_base,remote_base+7))+[local_base+20])
        rank_owner[rank].append([rank]*8)
    for repeat in (1,5):
        x=np.tile(np.array(rank_rows,np.int32),(1,repeat,1))
        owner=np.tile(np.array(rank_owner,np.int32),(1,repeat,1))
        case=save_case(out,f'packages80_repeat{repeat}',x,owner,x.shape[0]*x.shape[1],31)
        assert case['diagnostics'][0]==80
        cases.append(case)
    (out/'manifest.json').write_text(json.dumps({'scope':'synthetic correctness edge controls only','cases':cases},indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'max_packages':max(c['diagnostics'][0] for c in cases),
                      'max_samples':max(c['samples'] for c in cases)},indent=2))


if __name__=='__main__':main()
