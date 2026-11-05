"""CPU proof check: tiled C1 reuse prefix equals a new stable (expert,owner) sort.

Checks every task of both folded sources in eight real layers, plus arbitrary
binary reassignment. No GPU timing; this isolates the prefix arithmetic.
"""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def reuse_ordinals(routes,owner,block=256):
    route=routes.ravel();own=owner.ravel();order=np.argsort(route,kind='stable')
    e=route[order];flags=(own[order]==1).astype(np.int32);M=len(e)
    local=np.zeros(M,np.int32);summary=[];last=[]
    for begin in range(0,M,block):
        end=min(M,begin+block);keys=e[begin:end];v=flags[begin:end]
        heads=np.r_[begin==0 or e[begin]!=e[begin-1],keys[1:]!=keys[:-1]]
        acc=np.cumsum(v,dtype=np.int32);reset=np.maximum.accumulate(np.where(heads,np.arange(len(v)),0))
        before=np.r_[0,acc[:-1]]
        local[begin:end]=acc-before[reset]
        summary.append((int(local[end-1]),bool(heads.any())));last.append(int(keys[-1]))
    scanned=[];running=0
    for value,head in summary:
        running=value if head else running+value;scanned.append(running)
    ones=local.copy()
    for b in range(1,len(summary)):
        sl=slice(b*block,min(M,(b+1)*block));ones[sl]+=np.where(e[sl]==last[b-1],scanned[b-1],0)
    counts=np.bincount(route,minlength=352);starts=np.cumsum(counts)-counts
    occurrence=np.arange(M)-starts[e]
    ordinal=np.where(flags,ones-1,occurrence-ones)
    out=np.empty(M,np.int32);out[order]=ordinal
    # Independent oracle starts from original tasks and sorts by the final key.
    key=route*2+own;full_order=np.argsort(key,kind='stable')
    final_counts=np.bincount(key,minlength=704)
    expected=np.empty(M,np.int32)
    expected[full_order]=np.arange(M)-np.repeat(np.cumsum(final_counts)-final_counts,final_counts)
    np.testing.assert_array_equal(out,expected)
    return int(counts.max())


def main():
    rng=np.random.default_rng(20261105);rows=[]
    for layer in range(8):
        f=ROOT/f'outputs/two_gpu_overlap/inputs/layer_{layer:02}.npz';z=np.load(f)
        for rank in range(2):
            routes=z['topk'][rank];seed=z['owner'][rank]
            maximum=reuse_ordinals(routes,seed)
            arbitrary=rng.integers(0,2,size=seed.shape,dtype=np.int32)
            reuse_ordinals(routes,arbitrary)
            rows.append(dict(layer=layer,rank=rank,pairs=int(routes.size),native_assignment=True,arbitrary_assignment=True,max_expert_pairs=maximum,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    output=ROOT/'experiments/two_gpu_overlap/moonep_reuse_reference_check_result.json'
    output.write_text(json.dumps(dict(status='passed',cases=32,checked_pairs=sum(r['pairs'] for r in rows)*2,rows=rows),indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
