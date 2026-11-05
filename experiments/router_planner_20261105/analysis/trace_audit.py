"""Restricted router-dump decoding and reproducible structural analysis.

Only the explicitly supported torch CPU IntStorage tensor pickle is accepted.
No torch installation and no arbitrary pickle global import are used.
"""
from pathlib import Path
from collections import OrderedDict
import io, pickle, pickletools, json, hashlib, struct, argparse, importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXPERTS=352
RANKS=8

class StorageType:
    pass

class InnerUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('torch','IntStorage'):
            return StorageType
        raise pickle.UnpicklingError(f'Unsupported inner global: {module}.{name}')
    def persistent_load(self,pid):
        if not (isinstance(pid,tuple) and len(pid)==6 and pid[0]=='storage' and pid[1] is StorageType and pid[3]=='cpu' and isinstance(pid[4],int) and pid[5] is None):
            raise pickle.UnpicklingError(f'Unsupported storage metadata: {pid}')
        return pid

def load_storage(blob):
    f=io.BytesIO(blob)
    magic=InnerUnpickler(f).load()
    version=InnerUnpickler(f).load()
    sysinfo=InnerUnpickler(f).load()
    pid=InnerUnpickler(f).load()
    keys=InnerUnpickler(f).load()
    if magic!=119547037146038801333356 or version!=1001 or not sysinfo['little_endian'] or keys != [pid[2]]:
        raise pickle.UnpicklingError('Unsupported legacy torch storage framing')
    count=struct.unpack('<q',f.read(8))[0]
    data=f.read()
    if count!=pid[4] or count<0 or len(data)!=4*count:
        raise pickle.UnpicklingError('Storage size mismatch')
    return np.frombuffer(data,dtype='<i4').copy()

def rebuild_tensor(storage,offset,shape,stride,requires_grad,hooks):
    if not (isinstance(storage,np.ndarray) and storage.ndim==1 and offset==0 and len(shape)==2 and stride==(shape[1],1) and np.prod(shape)==storage.size and requires_grad is False and not hooks):
        raise pickle.UnpicklingError('Unsupported tensor shape/storage layout')
    return storage.reshape(shape).copy()

class RouterUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('torch._utils','_rebuild_tensor_v2'):rebuild_tensor, ('torch.storage','_load_from_bytes'):load_storage, ('collections','OrderedDict'):OrderedDict}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'Unsupported global {module}.{name}')
        return allowed[(module,name)]

def load_router(path):
    blob=Path(path).read_bytes()
    # Parse every outer opcode first; genops validates framing without execution.
    list(pickletools.genops(blob))
    obj=RouterUnpickler(io.BytesIO(blob)).load()
    if not isinstance(obj,np.ndarray):raise ValueError('Not a tensor')
    return obj

def js_divergence_rows(p):
    p=np.asarray(p,dtype=np.float64);p=p/p.sum(axis=-1,keepdims=True)
    avg=p.mean(axis=0)
    return float(np.sum(np.where(p>0,p*np.log2(np.maximum(p,1e-300)/np.maximum(avg,1e-300)),0))/len(p))

def remote_destinations(topk,home):
    destinations=home[topk]
    out=np.zeros((topk.shape[0],RANKS),dtype=np.int64)
    hist=np.zeros(RANKS+1,dtype=np.int64)
    for r in range(topk.shape[0]):
        hits=np.stack([(destinations[r]==d).any(axis=1) for d in range(RANKS)],axis=1)
        hits[:,r]=False
        out[r]=hits.sum(axis=0)
        hist+=np.bincount(hits.sum(axis=1),minlength=RANKS+1)
    return out,hist

def cooccurrence(topk):
    flat=topk.reshape(-1,topk.shape[-1])
    matrix=np.zeros((EXPERTS,EXPERTS),dtype=np.int64)
    for i in range(flat.shape[1]):
        for j in range(i+1,flat.shape[1]):
            counts=np.bincount(flat[:,i]*EXPERTS+flat[:,j],minlength=EXPERTS**2).reshape(EXPERTS,EXPERTS)
            matrix+=counts+counts.T
    return matrix

def rewire_within_source(topk,seed=20261026,sweeps=16):
    """Constrained swaps preserve each source/slot/expert count and row uniqueness.

    This finite Markov-chain control is not claimed to be an independent sample
    or to have mixed to a uniform distribution. It weakens within-token structure
    while retaining exact marginal counts, the source labels, and legal top-k.
    """
    result=topk.copy();rng=np.random.default_rng(seed);accepted=0;attempted=0
    for sweep in range(sweeps):
        for r in range(result.shape[0]):
            for slot in rng.permutation(result.shape[-1]):
                order=rng.permutation(result.shape[1]);a=order[:len(order)//2];b=order[len(order)//2:len(order)//2*2]
                ea=result[r,a,slot].copy();eb=result[r,b,slot].copy()
                ok=~np.any(result[r,a]==eb[:,None],axis=1)&~np.any(result[r,b]==ea[:,None],axis=1)
                result[r,a[ok],slot]=eb[ok];result[r,b[ok],slot]=ea[ok]
                accepted+=int(ok.sum());attempted+=len(ok)
    assert np.all(np.diff(np.sort(result,axis=-1),axis=-1)>0)
    for r in range(result.shape[0]):
        for slot in range(result.shape[-1]):
            assert np.array_equal(np.bincount(topk[r,:,slot],minlength=EXPERTS),np.bincount(result[r,:,slot],minlength=EXPERTS))
    return result,{'sweeps':sweeps,'seed':seed,'accepted_swaps':accepted,'attempted_swaps':attempted,'exact_source_slot_marginals':True,'unique_experts_per_token':True}

def summarize(topk,label,outdir,valid_control=False):
    r,n,k=topk.shape;t=r*n
    counts=np.stack([np.bincount(a.ravel(),minlength=EXPERTS) for a in topk])
    demand=counts.sum(axis=0);sorted_ids=np.argsort(-demand,kind='stable')
    home=np.arange(EXPERTS,dtype=np.int32)//(EXPERTS//RANKS)
    home_load=np.bincount(home,weights=demand,minlength=RANKS).astype(np.int64)
    destinations,dest_hist=remote_destinations(topk,home)
    co=cooccurrence(topk)
    upper=np.triu_indices(EXPERTS,1)
    pair_ids=np.argsort(-co[upper],kind='stable')[:20]
    # Correlation diagnostic only: demand-preserving slot shuffle, independently within source rank.
    # This exactly preserves rank marginal and top-k-slot counts, but can create duplicate IDs in a token;
    # report duplicates and use only as a cooccurrence dependence null, never as a valid planner input.
    rng=np.random.default_rng(20261026)
    shuffled=topk.copy()
    for rank in range(r):
        for j in range(k):rng.shuffle(shuffled[rank,:,j])
    nullco=cooccurrence(shuffled)
    total_pair_mass=float(co[upper].sum())
    pair_tv=float(0.5*np.abs(co[upper]-nullco[upper]).sum()/total_pair_mass)
    empirical_top20=co[upper][pair_ids].sum()/total_pair_mass
    null_on_top20=nullco[upper][pair_ids].sum()/total_pair_mass
    source_probs=counts/counts.sum(axis=1,keepdims=True)
    global_probs=demand/demand.sum()
    rank_tv=0.5*np.abs(source_probs-global_probs).sum(axis=1)
    unique,uc=np.unique(np.sort(topk.reshape(t,k),axis=1),axis=0,return_counts=True)
    stats={
      'label':label,'shape':list(topk.shape),'global_tokens':t,'pairs':t*k,'num_experts':EXPERTS,'top_k':k,
      'id_min':int(topk.min()),'id_max':int(topk.max()),'observed_experts':int(np.count_nonzero(demand)),
      'unique_experts_per_token':bool(np.all(np.diff(np.sort(topk,axis=-1),axis=-1)>0)),
      'expert_max_over_mean':float(demand.max()/demand.mean()),'expert_max_count':int(demand.max()),'expert_min_count':int(demand.min()),
      'expert_effective_count_entropy':float(np.exp(-np.sum(global_probs[global_probs>0]*np.log(global_probs[global_probs>0])))),
      'home_assumption':'home[e]=e//44; verify with production expert map; dump has no placement metadata',
      'home_loads':home_load.tolist(),'home_max_over_mean':float(home_load.max()/home_load.mean()),
      'home_max_minus_min_over_mean':float((home_load.max()-home_load.min())/home_load.mean()),
      'source_rank_js_bits':js_divergence_rows(counts),'source_rank_total_variation_from_global':rank_tv.tolist(),
      'top_experts':[{'id':int(e),'count':int(demand[e]),'token_frequency':float(demand[e]/t)} for e in sorted_ids[:24]],
      'top_expert_pair_share':{str(m):float(demand[sorted_ids[:m]].sum()/demand.sum()) for m in [8,12,16,20,24,32,64]},
      'top_expert_count_per_token':{str(m):np.bincount(np.isin(topk,sorted_ids[:m]).sum(axis=-1).ravel(),minlength=k+1).tolist() for m in [8,16,24]},
      'home_remote_destinations_total':int(destinations.sum()),'home_remote_destinations_per_token':float(destinations.sum()/t),
      'home_remote_destination_histogram':dest_hist.tolist(),'home_remote_tokens_by_source_destination':destinations.tolist(),
      'distinct_unordered_topk_sets':int(len(unique)),'most_repeated_unordered_topk_set_count':int(uc.max()),
      'unordered_topk_set_collision_fraction':float((t-len(unique))/t),
      'cooccurrence_pair_tv_vs_slot_shuffle':pair_tv,
      'cooccurrence_top20_mass':float(empirical_top20),'cooccurrence_slot_shuffle_mass_on_real_top20':float(null_on_top20),
      'slot_shuffle_duplicate_token_fraction':float(np.mean(np.any(np.diff(np.sort(shuffled,axis=-1),axis=-1)==0,axis=-1))),
      'cooccurrence_top20':[{'a':int(upper[0][q]),'b':int(upper[1][q]),'count':int(co[upper][q]),'slot_shuffle_count':int(nullco[upper][q])} for q in pair_ids],
      'sha256_topk_int32_c':hashlib.sha256(topk.tobytes()).hexdigest(),
    }
    if valid_control:
        rewired,rewire_meta=rewire_within_source(topk)
        rewired_co=cooccurrence(rewired)
        stats['cooccurrence_pair_tv_vs_legal_rewire']=float(0.5*np.abs(co[upper]-rewired_co[upper]).sum()/total_pair_mass)
        stats['cooccurrence_legal_rewire_mass_on_real_top20']=float(rewired_co[upper][pair_ids].sum()/total_pair_mass)
        stats['legal_rewire']=rewire_meta
        rank_shuffled=topk.reshape(t,k)[np.random.default_rng(20261026).permutation(t)].reshape(topk.shape)
        np.savez_compressed(outdir/f'{label}_rewired.npz',topk=rewired,home=home,num_experts=np.asarray(EXPERTS),weights_present=np.asarray(False))
        np.savez_compressed(outdir/f'{label}_source_shuffled.npz',topk=rank_shuffled,home=home,num_experts=np.asarray(EXPERTS),weights_present=np.asarray(False))
    np.savez_compressed(outdir/f'{label}_statistics.npz',expert_counts_by_source=counts,expert_demand=demand,cooccurrence=co,slot_shuffle_cooccurrence=nullco,home=home)
    return stats

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/trace_audit');ap.add_argument('--synthetic',action='store_true');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[];files=[]
    for layer in range(8):
        paths=[ROOT/f'inputs/iteration_4/microbatch_2/rank_{r:02d}/layer_{layer:02d}.pkl' for r in range(8)]
        topk=np.stack([load_router(p) for p in paths]).astype(np.int32,copy=False)
        assert topk.ndim==3 and topk.shape[0]==8 and 0<=topk.min() and topk.max()<EXPERTS
        name=f'layer_{layer:02d}'
        np.savez_compressed(args.output/f'{name}.npz',topk=topk,home=np.arange(EXPERTS,dtype=np.int32)//44,source_rank=np.broadcast_to(np.arange(8,dtype=np.int32)[:,None],topk.shape[:2]).copy(),iteration=np.asarray(4),microbatch=np.asarray(2),layer=np.asarray(layer),num_experts=np.asarray(EXPERTS),weights_present=np.asarray(False))
        records.append(summarize(topk,name,args.output,valid_control=True))
        files.extend([{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
        print(name,{k:records[-1][k] for k in ['shape','expert_max_over_mean','home_max_over_mean','home_remote_destinations_per_token','source_rank_js_bits','cooccurrence_pair_tv_vs_slot_shuffle']},flush=True)
    report={'schema_version':1,'dataset':'iteration_4/microbatch_2','layers':records,'files':files,'limitations':['Only one iteration and one microbatch; no temporal stability evidence','Only int32 top-k expert IDs; no gate weights, logits, model identity, sequence lengths, padding mask, hidden dimension, dtype, or physical placement metadata','home[e]=e//44 follows MoonEP contiguous-home convention and is an explicit assumption','Source/token/top-k order preserved; token positions are not automatically semantic sequence positions','Slot-shuffled routes are diagnostic controls only and can contain duplicate expert IDs']}
    if args.synthetic:
        spec=importlib.util.spec_from_file_location('hotcore_scenarios',Path('/Users/liangzuning/Documents/dev_env/EP/experiments/moe_placement/hotcore128k/scenarios.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        synthetic=[]
        for core,layout in [(8,'concentrated'),(16,'concentrated'),(24,'concentrated'),(16,'spread'),(16,'random')]:
            routes,home,meta=mod.generate_scenario(core_count=core,layout=layout)
            name=f'synthetic_c{core}_{layout}'
            stats=summarize(routes,name,args.output);stats['generator_metadata']={k:meta[k] for k in ['tokens','top_k','core_count','layout','realized_core_per_token','expert_max_over_mean','home_max_over_mean']};synthetic.append(stats)
            print(name,{k:stats[k] for k in ['expert_max_over_mean','home_max_over_mean','home_remote_destinations_per_token','source_rank_js_bits','cooccurrence_pair_tv_vs_slot_shuffle']},flush=True)
        report['synthetic_comparison']=synthetic
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
