"""Replay historical solver-free strategies on the supplied native MoonEP seed.

CPU semantic comparison. Sharded mode models independent complete-token shards;
its serial wall time is deliberately not a multi-GPU performance estimate.
"""
from pathlib import Path
import sys, time, json, argparse, hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from moonep_cpu import plan, communication
from vendor.residency import select_support
from vendor.bucket_optimizer import optimize
from vendor.bucket_sharded import _run_round

ROOT=Path(__file__).resolve().parents[1]

def sharded(x,seed,support,rounds=4):
    R,S,K=x.shape
    resident=np.sum(support.astype(np.int64)*(1<<np.arange(R))[:,None],axis=0)
    result=seed.copy(); predicted=0
    for rank in range(R):
        local=np.ascontiguousarray(result[rank],dtype=np.int32)
        for step in range(rounds):
            out=_run_round(np.ascontiguousarray(x[rank]),local,resident,R,S,1,False,step,rank*S)
            local=out[0];predicted+=int(out[10])
        result[rank]=local
    return result,{'predicted_gain':predicted,'rounds':rounds,'physical_gpus_used':0,
                   'scope':'CPU semantics of independent source-rank shards'}

def audit(x,seed,dst,support,budget):
    R,S,K=x.shape;E=support.shape[1]; home=np.arange(E)//(E//R)
    assert dst.shape==x.shape and np.all((0<=dst)&(dst<R))
    assert np.all(support[dst,x]),'execution without residency'
    before=communication(x,seed,E);after=communication(x,dst,E)
    assert before['pair_loads']==after['pair_loads'],'load changed'
    extras=(support&(home[None,:]!=np.arange(R)[:,None])).sum(1)
    assert int(extras.max())<=budget,'slot overflow'
    assert after['remote_token_destinations']<=before['remote_token_destinations']
    return after,extras.tolist()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layers',nargs='+',type=int,default=list(range(8)))
    ap.add_argument('--budgets',nargs='+',type=int,default=[8,44])
    ap.add_argument('--output',default='outputs/replay/strategies.json');a=ap.parse_args()
    output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Warm only a small representative shape; all measurements remain CPU semantics.
    for layer in a.layers:
        inp=ROOT/f'outputs/trace_audit/layer_{layer:02d}.npz'
        x=np.load(inp)['topk'].astype(np.int32);home=np.arange(352,dtype=np.int32)//44
        for budget in a.budgets:
            t=time.perf_counter();p=plan(x,E=352,B=budget);seed_time=time.perf_counter()-t
            base=communication(x,p.owner,352)
            row={'layer':layer,'budget':budget,'shape':list(x.shape),'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
                 'baseline':base,'cpu_seed_seconds':seed_time,'baseline_unsupported_pairs':int((p.physical_topk<0).sum()),'variants':[]}
            if (p.physical_topk<0).any():
                row['infeasible']='native allocation exceeds requested B';rows.append(row);continue
            for method in ('seed','visited','reciprocal'):
                start=time.perf_counter()
                support,diag=select_support(x,p.owner,home,slots=budget,method=method,sample_tokens=4096,groups_per_direction=4,seed=20261026)
                support_seconds=time.perf_counter()-start
                for mode in ('global','source_sharded'):
                    start=time.perf_counter()
                    if mode=='global':dst,details=optimize(x,p.owner,support,rounds=4,mode='bundle')
                    else:dst,details=sharded(x,p.owner,support)
                    seconds=time.perf_counter()-start
                    metrics,extras=audit(x,p.owner,dst,support,budget)
                    saved=base['remote_token_destinations']-metrics['remote_token_destinations']
                    if mode=='source_sharded':assert saved==details['predicted_gain']
                    name=f'{method}_{mode}_4'
                    v={'name':name,'metrics':metrics,'reserved_extra_experts':extras,'remote_saved':saved,
                       'remote_reduction_percent':100*saved/base['remote_token_destinations'],
                       'N_all':x.size/metrics['all_token_destinations'],
                       'active_replica_delta':sum(metrics['active_extra_experts'])-sum(base['active_extra_experts']),
                       'cpu_support_seconds':support_seconds,'cpu_exchange_seconds':seconds,
                       'diagnostics':diag,'exchange_details':details,'audit_pass':True}
                    row['variants'].append(v)
                    # Compact placements for independent GPU-quality checks and reuse.
                    np.savez_compressed(output.parent/f'layer{layer:02d}_b{budget}_{name}.npz',owner=dst,support=support)
                    print(json.dumps({'layer':layer,'B':budget,'variant':name,'reduction_pct':round(v['remote_reduction_percent'],4),
                                      'extra_active':sum(metrics['active_extra_experts']),'cpu_seconds':round(seconds+support_seconds,3)}),flush=True)
            rows.append(row)
            output.write_text(json.dumps({'scope':'CPU semantic replay; no GPU or end-to-end timing','rows':rows},indent=2)+'\n')

if __name__=='__main__':main()
