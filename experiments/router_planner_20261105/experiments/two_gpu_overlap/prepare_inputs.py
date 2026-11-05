"""Pin a two-physical-rank replay without inventing new expert choices.

Adjacent four original source shards become one rank. Expert homes are e//176.
This is a derived two-rank topology, not an original two-rank production dump.
"""
from pathlib import Path
import hashlib,json,sys,time,types
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
try:
    import numba
except ImportError:
    # reciprocal selection uses ordinary Python; its optional njit helpers are
    # never invoked in this script. Keep the original source unmodified.
    stub=types.ModuleType('numba')
    stub.njit=lambda *a,**kw: a[0] if a and callable(a[0]) else lambda f:f
    sys.modules['numba']=stub
from moonep_cpu import plan,communication
from vendor.residency import select_support

def main():
    out=ROOT/'outputs/two_gpu_overlap/inputs';out.mkdir(parents=True,exist_ok=True)
    manifest=dict(topology='derived: original source ranks 0..3 -> physical 0; 4..7 -> physical 1',
                  experts=352,topk=8,extra_slots=8,sample_tokens=256,groups_per_class=2,
                  sample_seed=20261015,weights_in_trace=False,layer_records=[])
    for layer in range(8):
        f=ROOT/f'outputs/trace_audit/layer_{layer:02}.npz'
        original=np.load(f,allow_pickle=False)['topk'];x=original.reshape(2,-1,8).copy()
        tick=time.perf_counter();seed=plan(x,E=352,B=8,replica_target=0);cpu_seed_s=time.perf_counter()-tick
        assert (seed.physical_topk>=0).all()
        tick=time.perf_counter()
        support,diag=select_support(x,seed.owner,np.arange(352)//176,slots=8,
            method='reciprocal',sample_tokens=256,groups_per_direction=2,seed=20261015)
        cpu_select_s=time.perf_counter()-tick
        ids=np.sort(np.random.default_rng(20261015).choice(x.shape[0]*x.shape[1],256,replace=False))
        target=out/f'layer_{layer:02}.npz'
        np.savez_compressed(target,topk=x,owner=seed.owner.astype(np.int32),
            alloc=seed.alloc,tpe=seed.tpe.astype(np.int32),
            native_experts_to_copy=seed.experts_to_copy,support=support,
            sample_ids=ids.astype(np.int64),sample_routes=x.reshape(-1,8)[ids])
        row=dict(layer=layer,original_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            replay_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),shape=list(x.shape),
            cpu_seed_seconds=cpu_seed_s,cpu_select_seconds=cpu_select_s,
            native=communication(x,seed.owner),residency=diag)
        manifest['layer_records'].append(row)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(layer,row['shape'],diag['mandatory_extra'],diag['candidate_extra'],flush=True)

if __name__=='__main__':main()
