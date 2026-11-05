"""Two-H200 training communication replay, one microbatch per weight update.

Real BF16 expert copies, forward/backward token traffic, and native FP32
replica-gradient reduction. No expert GEMM, gradient generation, activation
retention, optimizer, or router derivative is simulated as model compute.
Known synthetic payloads audit the real communication. Separate from the
previous forward-only benchmark; never overwrites its evidence.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, sys, traceback
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'inputs/MoonEP'))
from benchmark import gather_object,timed
from moonep_common import create_buffer,compile_native,allocate_weights
from moonep_split import SplitPhaseMoonEP
from moonep_materialize_reuse import CudaMoonEPMaterializerReuse
from exchange_cuda import TwoRankExchange
from residency_cuda_fused import TwoRankResidency
from training_grad import allocate_gradients


def set_weight_version(pool,version):
    """Untimed stand-in for updated source weights; adjacent versions differ.

    Seven repeating small integer patterns keep all audit values exactly BF16.
    Version IDs are unique; this is not an optimizer or a learned checkpoint.
    Copy slots are invalidated before EACH timed full call, preventing hot reuse.
    """
    epn=pool.E//pool.R;lo=pool.rank*epn;hi=lo+epn;phase=version%7
    ids=torch.arange(lo,hi,device=pool.W1.device,dtype=torch.int32)
    a=(ids%31+1+phase*2).to(torch.bfloat16).view(epn,1,1)
    b=(ids//31+1+phase*2).to(torch.bfloat16).view(epn,1,1)
    pool.W1[lo:hi].copy_(a.expand_as(pool.W1[lo:hi]))
    pool.W2[lo:hi].copy_(b.expand_as(pool.W2[lo:hi]))
    pool.W1[pool.E:].fill_(-1);pool.W2[pool.E:].fill_(-1)


def check_weight_version(pool,plan,version):
    phase=version%7;count=0
    for slot,expert in enumerate(plan.experts_to_copy[pool.rank].cpu().tolist()):
        if expert<0:continue
        assert bool((pool.W1[pool.E+slot]==expert%31+1+phase*2).all()),'stale/corrupt W1 replica'
        assert bool((pool.W2[pool.E+slot]==expert//31+1+phase*2).all()),'stale/corrupt W2 replica'
        count+=1
    return dict(version_id=version,pattern_phase=phase,active_full_slots_checked=count)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--layers',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--hidden',type=int,default=7168)
    ap.add_argument('--intermediate',type=int,default=2048)
    ap.add_argument('--rounds',type=int,default=1)
    ap.add_argument('--samples',type=int,default=10)
    ap.add_argument('--warmup',type=int,default=2)
    ap.add_argument('--num-sms',type=int,default=32)
    ap.add_argument('--prefetch-sms',type=int,default=32)
    ap.add_argument('--mode',choices=['eager','graph'],default='graph')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    assert dist.get_world_size()==2
    if args.output.exists():raise FileExistsError('Use a new attempt path; prior evidence is retained')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    E,B,K=352,8,8;param_count=3*args.hidden*args.intermediate
    record=dict(status='running',metadata=dict(
        training=True,microbatches_per_update=1,weight_refresh_every_update=True,
        rank_count=2,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
        hidden=args.hidden,intermediate=args.intermediate,experts=E,extra_slots_per_rank=B,topk=K,
        replica_target=0,replica_max=7,enable_pdl=False,rounds=args.rounds,
        weight_dtype='BF16',gradient_dtype='FP32',activation_dtype='BF16',router_scalar_dtype='FP32',
        expert_parameter_count=param_count,weight_bytes_per_expert=param_count*2,
        gradient_bytes_per_expert=param_count*4,
        residency='fused_warp',materializer='reuse',num_sms=args.num_sms,prefetch_sms=args.prefetch_sms,
        samples=args.samples,warmup=args.warmup,timing_mode=args.mode,
        source_mapping='original source shards0..3->rank0 and4..7->rank1; all routes retained',
        scope='M=1 planning + BF16 expert copies + forward x/y + backward dY/dX + FP32 replica-gradient reduction',
        excluded='expert GEMM, actual gradient generation, activation retention, optimizer, data/model parallel collectives, graph/JIT/allocation/setup',
        synthetic_payloads='unique per-token integer hidden/dY, known router scalars, projection/expert/rank-specific FP32 gradients',
        router_transport='forward scalar weights and backward synthetic scalar gradients included; generation represented by a counted scalar-buffer copy',
        preparation='local tpe and fixed random sample IDs provided; reset gradients and update source weights outside timing; invalidate copy slots before every full sample',
        safety='single in-flight microbatch per buffer; saved forward plan/dedup retained through backward; gradient ready publish and reduction serialized with token communication',
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.cu')}
        ),layers=[],failures=[])
    def save():
        if rank==0:
            args.output.write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(dict(status=record['status'],layers=len(record['layers']),failures=len(record['failures']))),flush=True)
    save();buffer=split=weights=grads=None;graphs={};version=0
    try:
        for layer in map(int,args.layers.split(',')):
            path=ROOT/f'outputs/two_gpu_overlap/inputs/layer_{layer:02}.npz'
            data=np.load(path,allow_pickle=False);S=data['topk'].shape[1]
            def gpu(x,dtype=torch.int32):return torch.tensor(x,device='cuda',dtype=dtype)
            routes=gpu(data['topk'][rank]);seed=gpu(data['owner'][rank]);tpe=gpu(data['tpe'][rank])
            support_ref=gpu(data['support'],torch.bool)
            if buffer is None:
                buffer=create_buffer(S,args.hidden,num_sms=args.num_sms)
                compile_native(buffer,prefetch_sms=args.prefetch_sms,weight_shapes=[
                    (E,2*args.intermediate,args.hidden,B,torch.bfloat16),
                    (E,args.hidden,args.intermediate,B,torch.bfloat16)])
                split=SplitPhaseMoonEP(buffer,prefetch_sms=args.prefetch_sms)
                weights=allocate_weights(buffer,args.hidden,args.intermediate)
                grads=allocate_gradients(buffer,args.hidden,args.intermediate,layout='fused')
                grads.precompile()
                hidden=(torch.arange(S,device='cuda')%11+rank*13).to(torch.bfloat16)[:,None].expand(S,args.hidden).contiguous()
                dy=(torch.arange(S,device='cuda')%13+rank*17).to(torch.bfloat16)[:,None].expand(S,args.hidden).contiguous()
                router=((torch.arange(S*K,device='cuda')+rank*S*K)%31+1).float().reshape(S,K)
                saved_router=torch.empty(int(buffer._require_ctx()['NvS']),device='cuda',dtype=torch.float32)
                expected_dx=dy*K
                dist.barrier();torch.cuda.synchronize()
            mat=CudaMoonEPMaterializerReuse(buffer,routes)
            res=TwoRankResidency(gpu(data['sample_routes']),gpu(data['sample_ids'],torch.int64),
                                local_tokens=S,rank=rank,slots=B,variant='fused_warp')
            exchange=TwoRankExchange(routes,support_ref,rank=rank,rounds=args.rounds,variant='prefix')
            dist.barrier()
            def native():return split.plan_native(routes,tpe)
            def optimized(early=False):
                native();owner=mat.decode_native_owner(split.plan)
                support=res.run(split.plan.experts_to_copy,owner,local_routes=routes)
                copies,ready=mat.prepare_support(support)
                w=split.prefetch(copies,ready=ready,**weights.kwargs()) if early else None
                final=exchange.run(owner,support=support)
                p,cu,ready=mat.run(final)
                return p,cu,ready,w

            native();torch.cuda.synchronize()
            assert torch.equal(mat.decode_native_owner(split.plan),seed)
            assert torch.equal(split.plan.experts_to_copy,gpu(data['native_experts_to_copy']))
            p,cu,ready,_=optimized(False);torch.cuda.synchronize();mat.check_errors()
            assert torch.equal(res.support,support_ref)
            loads=torch.bincount(exchange.owner.flatten().long(),minlength=2);dist.all_reduce(loads)
            assert loads.tolist()==[S*K,S*K]
            assert bool(support_ref[exchange.owner.long(),routes.long()].all())
            remote_before=int((seed!=rank).any(1).sum());remote_after=int((exchange.owner!=rank).any(1).sum())
            before=gather_object(remote_before);after=gather_object(remote_after)
            copies_native=(split.plan.experts_to_copy>=0).sum(1).cpu().tolist()
            copies_opt=(mat.plan.experts_to_copy>=0).sum(1).cpu().tolist()
            row=dict(layer=layer,shape=list(data['topk'].shape),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     communication=dict(remote_before_by_rank=before,remote_after_by_rank=after,
                     native_copies=copies_native,optimized_copies=copies_opt,
                     activation_bytes_saved_per_microbatch=4*args.hidden*2*(sum(before)-sum(after)),
                     incremental_weight_bytes_per_update=(sum(copies_opt)-sum(copies_native))*param_count*2,
                     incremental_gradient_bytes_per_update=(sum(copies_opt)-sum(copies_native))*param_count*4),
                     correctness=dict(native_cpu=True,sampled_support_cpu=True,balanced=True,residency_legal=True),
                     measurements={},audits=[],rounds=exchange.stats.cpu().tolist())
            record['layers'].append(row);save()

            def training(is_opt,early,audit_forward=False):
                if is_opt:p,cu,r,w=optimized(early)
                else:
                    p,cu,r=native()
                    w=split.prefetch(p.experts_to_copy,ready=r,**weights.kwargs()) if early else None
                ticket=split.dispatch_fresh(hidden,router,plan=p,cu_seqlens=cu,ready=r)
                if w is None:w=split.prefetch(p.experts_to_copy,ready=ticket.done,**weights.kwargs())
                torch.cuda.current_stream().wait_event(ticket.done)
                torch.cuda.current_stream().wait_event(w)
                # Model computation is excluded. Preserve an independent scalar
                # gradient payload; no saved alias survives the next dispatch.
                saved_router.copy_(ticket.route_weights)
                forward=buffer.combine(plan=p,hidden_nvsh=ticket.hidden,zero_copy=True)[0]
                if audit_forward:torch.testing.assert_close(forward,hidden*K,rtol=0,atol=0)
                # Backward REUSES the exact forward plan and existing dedup map.
                back=buffer.dispatch(dy,plan=p,async_finish=True,zero_copy=True)
                torch.cuda.current_stream().wait_event(back[-1])
                dx,drouter,_=buffer.combine(plan=p,hidden_nvsh=back[0],route_weights_nvs=saved_router,
                                             zero_copy=True)
                done=grads.launch(p,publish=True)
                torch.cuda.current_stream().wait_event(done)
                return dict(dx=dx,drouter=drouter,plan=p)

            def grad_only(plan):
                done=grads.launch(plan,publish=True)
                torch.cuda.current_stream().wait_event(done)
                return dict(plan=plan)
            functions={
                'native_planner':native,'optimized_planner':lambda:optimized(False),
                'native_grad_reduce':lambda:grad_only(split.plan),
                'optimized_grad_reduce':lambda:grad_only(mat.plan),
                'native_training_serial':lambda:training(False,False),
                'native_training_early':lambda:training(False,True),
                'optimized_training_serial':lambda:training(True,False),
                'optimized_training_early':lambda:training(True,True)}
            def prepare(name):
                nonlocal_version=None
                if 'training' in name:
                    nonlocal_version=prepare.next_version;prepare.next_version+=1
                    set_weight_version(weights,nonlocal_version)
                if 'training' in name or 'grad_reduce' in name:
                    grads.reset(epoch=prepare.grad_epoch)
                    prepare.grad_epoch+=1
                return nonlocal_version
            prepare.next_version=layer*10000+1
            prepare.grad_epoch=0
            def audit(name,output,version):
                result={}
                if 'training' in name:
                    torch.testing.assert_close(output['dx'],expected_dx,rtol=0,atol=0)
                    torch.testing.assert_close(output['drouter'],router,rtol=0,atol=0)
                    result.update(backward_hidden_exact=True,backward_router_exact=True,
                                  weights=check_weight_version(weights,output['plan'],version))
                if 'training' in name or 'grad_reduce' in name:
                    result['gradients']=grads.check(output['plan'].experts_to_copy)
                if name.startswith('optimized'):mat.check_errors()
                return result

            # Validate both forward and backward while intermediate buffers
            # still contain their respective result. These checks are untimed.
            for opt in (False,True):
                label='optimized_training_early' if opt else 'native_training_early'
                v=prepare(label);dist.barrier();torch.cuda.synchronize()
                result=training(opt,True,audit_forward=True);torch.cuda.synchronize()
                row['audits'].append(dict(stage='eager_full',mode=label,per_rank=gather_object(audit(label,result,v))))
            for name,fn in functions.items():
                for _ in range(args.warmup):
                    prepare(name);dist.barrier();torch.cuda.synchronize();fn();torch.cuda.synchronize()
                row['measurements'][name]=[]
            graphs={};captured={}
            if args.mode=='graph':
                for name,fn in functions.items():
                    v=prepare(name);torch.cuda.synchronize();dist.barrier()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):captured[name]=fn()
                    graphs[name]=graph
                    graph.replay();torch.cuda.synchronize();dist.barrier()
                    row['audits'].append(dict(stage='graph_first_replay',mode=name,per_rank=gather_object(audit(name,captured[name],v))))
            names=list(functions)
            for sample in range(args.samples):
                order=names[sample%len(names):]+names[:sample%len(names)]
                for name in order:
                    v=prepare(name)
                    if args.mode=='graph':
                        callback=graphs[name].replay
                        measured=gather_object(timed(callback));output=captured[name]
                    else:
                        holder={}
                        def callback():holder['out']=functions[name]()
                        measured=gather_object(timed(callback));output=holder['out']
                    checks=gather_object(audit(name,output,v))
                    row['measurements'][name].append(dict(sample=sample,per_rank=measured,
                        max_cuda_ms=max(v['cuda_ms'] for v in measured),max_host_ms=max(v['host_ms'] for v in measured),
                        weight_version=v,per_rank_audits=checks))
                save()
            row['post_timing_checked']=True;save()
            callback=None;output=None;result=None
            for graph in graphs.values():graph.reset()
            graphs.clear();captured.clear();graph=None
            torch.cuda.synchronize();dist.barrier()
            del functions,mat,res,exchange,routes
        record['status']='complete';save()
    except BaseException as exc:
        record['status']='failed';record['failures'].append(dict(rank=rank,error=repr(exc),traceback=traceback.format_exc()));save();raise
    finally:
        for graph in graphs.values():graph.reset()
        graphs.clear()
        if split is not None:split.close()
        if grads is not None:grads.close()
        if weights is not None:weights.close()
        if buffer is not None:buffer.destroy()
        dist.destroy_process_group()


if __name__=='__main__':main()
