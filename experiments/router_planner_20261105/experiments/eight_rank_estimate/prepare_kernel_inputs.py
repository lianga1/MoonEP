"""Original R8 CPU oracle fixtures. No folding into physical GPU ranks."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from vendor.residency import select_support
from vendor.bucket_optimizer import _run_round,HAVE_NUMBA

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layers',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs/eight_rank_estimate/kernel_inputs')
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if not HAVE_NUMBA:raise RuntimeError('full input CPU oracle requires numba')
    records=[]
    for layer in map(int,args.layers.split(',')):
        out=args.output_dir/f'layer_{layer:02}.npz'
        if out.exists():raise FileExistsError(out)
        xp=ROOT/f'outputs/trace_audit/layer_{layer:02}.npz'
        sp=ROOT/f'outputs/rt2_b8_comparison/layer{layer:02}_native_rt0.npz'
        rp=ROOT/f'outputs/replay/layer{layer:02}_b8_small256_global_4.npz'
        z=np.load(xp);x=z['topk'].astype(np.int32);home=z['home']
        seedfile=np.load(sp);seed=seedfile['owner'].astype(np.int32)
        assert x.shape==(8,16768,8)
        start=time.perf_counter()
        support,diagnostics=select_support(x,seed,home,slots=8,method='reciprocal',sample_tokens=256,groups_per_direction=2,seed=20261026)
        cpu_residency=time.perf_counter()-start
        archived=np.load(rp)
        np.testing.assert_array_equal(support,archived['support'])
        resident=np.zeros(352,np.int64)
        for r in range(8):resident|=support[r].astype(np.int64)<<r
        owner=seed.reshape(-1,8).copy();prefixes=[seed.copy()];stats=[]
        start=time.perf_counter()
        for step in range(4):
            result=_run_round(x.reshape(-1,8),owner,resident,8,16768,1,False,step)
            owner=result[0];prefixes.append(owner.reshape(x.shape).copy())
            stats.append([result[6],result[8],result[9],result[10]])
        cpu_exchange=time.perf_counter()-start
        np.testing.assert_array_equal(prefixes[-1],archived['owner'])
        ids=np.sort(np.random.default_rng(20261026).choice(134144,256,replace=False)).astype(np.int64)
        with out.open('xb') as f:
            np.savez_compressed(f,topk=x,seed=seed,support=support,owner_prefixes=np.stack(prefixes),
                stats=np.array(stats,np.int64),native_experts_to_copy=seedfile['experts_to_copy'].astype(np.int32),
                sample_ids=ids)
        records.append(dict(layer=layer,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            original_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (xp,sp,rp)},
            cpu_residency_preparation_seconds=cpu_residency,cpu_exchange_oracle_seconds=cpu_exchange,
            cpu_times_are_not_gpu_estimates=True,archived_support_and_final_owner_exact=True,
            residency_diagnostics=diagnostics))
        print(json.dumps(dict(layer=layer,prepared=str(out),cpu_residency_s=cpu_residency,cpu_oracle_s=cpu_exchange)),flush=True)
    manifest=args.output_dir/'manifest.json'
    with manifest.open('x') as f:json.dump(dict(N=134144,R=8,S=16768,K=8,E=352,B=8,d=256,g=2,L=4,seed=20261026,layers=records),f,indent=2)

if __name__=='__main__':main()
