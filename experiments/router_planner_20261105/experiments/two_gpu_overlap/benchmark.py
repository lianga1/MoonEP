"""Paired, two physical GPU planner + real MoonEP payload/copy benchmark.

No expert GEMM: dispatch payloads are identity expert outputs and combine
performs the actual K-sum. Results describe the planning/communication segment,
not model latency. CPU references, allocations and JIT are outside timing.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,statistics,sys,time,traceback
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'inputs/MoonEP'))
from moonep_common import create_buffer,compile_native,allocate_weights,check_weight_slots
from moonep_split import SplitPhaseMoonEP
from moonep_materialize import CudaMoonEPMaterializer
from moonep_materialize_reuse import CudaMoonEPMaterializerReuse
from exchange_cuda import TwoRankExchange
from residency_cuda import TwoRankResidency


def gather_object(value):
    out=[None]*dist.get_world_size();dist.all_gather_object(out,value);return out


def timed(fn,repeats=1):
    dist.barrier();torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();wall=time.perf_counter()
    for _ in range(repeats):fn()
    b.record();b.synchronize()
    return dict(cuda_ms=a.elapsed_time(b)/repeats,host_ms=(time.perf_counter()-wall)*1000/repeats)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--layers',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--hidden',type=int,default=7168)
    ap.add_argument('--intermediate',type=int,default=2048)
    ap.add_argument('--rounds',type=int,default=4)
    ap.add_argument('--samples',type=int,default=10)
    ap.add_argument('--mode',choices=['eager','graph'],default='eager')
    ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--num-sms',type=int,default=32)
    ap.add_argument('--prefetch-sms',type=int,default=32)
    ap.add_argument('--timeline',action='store_true')
    ap.add_argument('--materializer',choices=['sort','reuse'],default='sort')
    ap.add_argument('--residency',choices=['base','tops','warp','combined','fused','fused_warp'],default='base')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--components-only',action='store_true')
    args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=12))
    assert dist.get_world_size()==2
    args.output.parent.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',metadata=dict(world=2,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
        hidden=args.hidden,intermediate=args.intermediate,weight_dtype='BF16',
        experts=352,extra_slots=8,topk=8,token_padding=128,
        replica_target=0,replica_max=7,enable_pdl=False,
        native_baseline='native planner with popularity copies disabled (rt=0); balancing copies retained',
        expert_bytes=3*args.hidden*args.intermediate*2,rounds=args.rounds,
        source_mapping='original 0..3 to rank0; 4..7 to rank1; all router rows retained',
        scope='planning + actual expert copies + actual dispatch/combine; identity experts, no GEMM',
        precomputed_inputs='local expert histogram and fixed random sample IDs; routes remain runtime GPU input',
        hot_scope='same trace/support and immutable version reused; no measured cross-step stability',
        diagnostic_scope='timeline/profiler invocations are untimed cold identity-expert diagnostics; kernel interval overlap is not NVLink byte occupancy',
        samples=args.samples,warmup=args.warmup,num_sms=args.num_sms,prefetch_sms=args.prefetch_sms,
        timing_mode=args.mode,hot_exact_map_prefill_per_sample=True,
        materializer=args.materializer,residency=args.residency,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.cu')}),
        layers=[],failures=[])
    def save():
        if rank==0:args.output.write_text(json.dumps(record,indent=2));print(json.dumps({'status':record['status'],'layers':len(record['layers']),'failures':len(record['failures'])}),flush=True)
    save();buffer=None;pool=None;split=None
    try:
        for layer in map(int,args.layers.split(',')):
            f=ROOT/f'outputs/two_gpu_overlap/inputs/layer_{layer:02}.npz'
            data=np.load(f,allow_pickle=False);S=data['topk'].shape[1]
            routes=torch.tensor(data['topk'][rank],dtype=torch.int32,device='cuda')
            seed_cpu=torch.tensor(data['owner'][rank],dtype=torch.int32,device='cuda')
            support_cpu=torch.tensor(data['support'],dtype=torch.bool,device='cuda')
            tpe=torch.tensor(data['tpe'][rank],dtype=torch.int32,device='cuda')
            if buffer is None:
                buffer=create_buffer(S,args.hidden,num_sms=args.num_sms)
                compile_native(buffer,prefetch_sms=args.prefetch_sms,
                    weight_shapes=((352,2*args.intermediate,args.hidden,8,torch.bfloat16),
                                   (352,args.hidden,args.intermediate,8,torch.bfloat16)))
                split=SplitPhaseMoonEP(buffer,prefetch_sms=args.prefetch_sms)
                if not args.components_only:
                    pool=allocate_weights(buffer,args.hidden,args.intermediate);pool.fill_sources()
                hidden=torch.full((S,args.hidden),float(rank+1),dtype=torch.bfloat16,device='cuda')
                dist.barrier();torch.cuda.synchronize()
            sort_mat=CudaMoonEPMaterializer(buffer,routes)
            reuse_mat=CudaMoonEPMaterializerReuse(buffer,routes)
            mat=reuse_mat if args.materializer=='reuse' else sort_mat
            dist.barrier()
            residency_cls=TwoRankResidency
            residency_options={}
            if args.residency in ('fused','fused_warp'):
                from residency_cuda_fused import TwoRankResidency as FusedResidency
                residency_cls=FusedResidency
                residency_options['variant']=args.residency
            elif args.residency!='base':
                from residency_cuda_optimized import TwoRankResidency as OptimizedResidency
                residency_cls=OptimizedResidency
                residency_options['variant']=args.residency
            res=residency_cls(torch.tensor(data['sample_routes'],dtype=torch.int32,device='cuda'),
                torch.tensor(data['sample_ids'],dtype=torch.int64,device='cuda'),local_tokens=S,rank=rank,slots=8,
                **residency_options)
            fast=TwoRankExchange(routes,support_cpu,rank=rank,rounds=args.rounds,variant='prefix')
            old=TwoRankExchange(routes,support_cpu,rank=rank,rounds=args.rounds,variant='sort')
            ctx=buffer._require_ctx()
            def native():
                return split.plan_native(routes,tpe)
            def decode():
                return mat.decode_native_owner(split.plan)
            native();torch.cuda.synchronize();native_owner=decode()
            assert torch.equal(native_owner,seed_cpu),'native owner differs CPU reference'
            assert torch.equal(split.plan.experts_to_copy,torch.tensor(data['native_experts_to_copy'],dtype=torch.int32,device='cuda')),'native copies differ'
            actual_support=res.run(split.plan.experts_to_copy,native_owner,local_routes=routes)
            assert torch.equal(actual_support,support_cpu),'GPU sampled residency differs CPU reference'
            a=fast.run(native_owner,support=actual_support).clone()
            b=old.run(native_owner,support=actual_support).clone()
            assert torch.equal(a,b),'prefix/sort owner mismatch'
            assert torch.equal(fast.stats,old.stats),'prefix/sort stats mismatch'
            mat.prepare_support(actual_support);plan,cu,ready=mat.run(a);mat.check_errors()
            source_start=rank*int(ctx['meta_chunk_padded'])+int(ctx['SRC_INFO_OFF'])
            fields=('dst','physical_topk','experts_to_copy','zero_fill_ranges','remote_stats')
            snapshot={key:getattr(plan,key).clone() for key in fields}
            snapshot['cu']=cu.clone();snapshot['source']=ctx['meta_buf'][source_start:source_start+int(ctx['NvS'])].clone()
            other_mat=sort_mat if args.materializer=='reuse' else reuse_mat
            other_mat.prepare_support(actual_support);other_plan,other_cu,_=other_mat.run(a);other_mat.check_errors()
            for key in fields:assert torch.equal(getattr(other_plan,key),snapshot[key]),'C1 reuse differs: '+key
            assert torch.equal(other_cu,snapshot['cu'])
            assert torch.equal(ctx['meta_buf'][source_start:source_start+int(ctx['NvS'])],snapshot['source'])
            local_before=int((native_owner!=rank).any(dim=1).sum().item())
            local_after=int((a!=rank).any(dim=1).sum().item())
            counts=torch.bincount(a.reshape(-1).long(),minlength=2);dist.all_reduce(counts)
            assert counts.tolist()==[S*8,S*8]
            legal=actual_support[a.long(),routes.long()].all().item();assert legal
            row=dict(layer=layer,input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),shape=list(data['topk'].shape),
                correctness=dict(native_cpu_owner=True,native_cpu_copies=True,sampled_residency_cpu=True,
                    sort_prefix_owner=True,sort_prefix_stats=True,balanced=True,residency_legal=True,materializers_exact=True),
                communication=dict(remote_before_by_rank=gather_object(local_before),remote_after_by_rank=gather_object(local_after),
                    native_copies=(split.plan.experts_to_copy>=0).sum(1).cpu().tolist(),
                    optimized_copies=(mat.plan.experts_to_copy>=0).sum(1).cpu().tolist(),
                    dispatch_combine_bytes_saved=2*args.hidden*2*sum(gather_object(local_before-local_after))),
                rounds=fast.stats.cpu().tolist(),measurements={})
            record['layers'].append(row);save()

            # Fixed-shape full pipeline: all support decisions run on GPU each time.
            def opt_plan(early_weights=False):
                native();owner=decode()
                support=res.run(split.plan.experts_to_copy,owner,local_routes=routes)
                etc,expert_ready=mat.prepare_support(support)
                weight_done=split.prefetch(etc,ready=expert_ready,**pool.kwargs()) if early_weights else None
                final=fast.run(owner,support=support)
                plan,cu,token_ready=mat.run(final)
                return plan,cu,token_ready,weight_done
            def finish(plan,cu,ready,weight_done=None,cold=False):
                ticket=split.dispatch_fresh(hidden,None,plan=plan,cu_seqlens=cu,ready=ready)
                if cold and weight_done is None:
                    weight_done=split.prefetch(plan.experts_to_copy,ready=ticket.done,**pool.kwargs())
                torch.cuda.current_stream().wait_event(ticket.done)
                if weight_done is not None:torch.cuda.current_stream().wait_event(weight_done)
                result=buffer.combine(plan=plan,hidden_nvsh=ticket.hidden,zero_copy=True)
                return result[0]
            def native_full(cold=False,early=False):
                plan,cu,ready=native()
                w=split.prefetch(plan.experts_to_copy,ready=ready,**pool.kwargs()) if cold and early else None
                return finish(plan,cu,ready,w,cold)
            def opt_full(cold=False,early=False):
                plan,cu,ready,w=opt_plan(early_weights=cold and early)
                return finish(plan,cu,ready,w,cold)
            def native_weight():
                w=split.prefetch(split.plan.experts_to_copy,ready=torch.cuda.current_stream().record_event(),**pool.kwargs())
                torch.cuda.current_stream().wait_event(w)
            def opt_weight():
                w=split.prefetch(mat.plan.experts_to_copy,ready=torch.cuda.current_stream().record_event(),**pool.kwargs())
                torch.cuda.current_stream().wait_event(w)
            # Component callbacks never modify the preserved native plan.
            functions={'native_planner':native,'sampled_residency':lambda:res.run(split.plan.experts_to_copy,decode(),local_routes=routes),
                'exchange_sort':lambda:old.run(seed_cpu,support=support_cpu),
                'exchange_prefix':lambda:fast.run(seed_cpu,support=support_cpu),
                'materialize':lambda:mat.run(a),'materialize_sort':lambda:sort_mat.run(a),
                'materialize_reuse':lambda:reuse_mat.run(a),'optimized_full_planner':lambda:opt_plan(False)}
            if not args.components_only:
                functions.update(native_weights=native_weight,optimized_weights=opt_weight,
                    native_cold_serial=lambda:native_full(True,False),native_cold_early=lambda:native_full(True,True),
                    native_hot=lambda:native_full(False,False),optimized_cold_serial=lambda:opt_full(True,False),
                    optimized_cold_early=lambda:opt_full(True,True),optimized_hot=lambda:opt_full(False,False))
                # Full payload validation, outside timing. All 8 identity outputs sum to 8*x.
                for label,fn in [('native',lambda:native_full(True,True)),('optimized',lambda:opt_full(True,True))]:
                    output=fn();torch.cuda.synchronize()
                    torch.testing.assert_close(output,hidden*8,rtol=0,atol=0)
                    check_weight_slots(pool,split.plan.experts_to_copy if label=='native' else mat.plan.experts_to_copy)
                    row['correctness'][label+'_payload_and_weights']=True
            for name,fn in functions.items():
                for _ in range(args.warmup):fn()
                torch.cuda.synchronize();dist.barrier()
                row['measurements'][name]=[]
            graphs={}
            if args.mode=='graph':
                for name,fn in functions.items():
                    torch.cuda.synchronize();dist.barrier()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):captured_output=fn()
                    graphs[name]=graph
                    graph.replay();torch.cuda.synchronize();dist.barrier()
                    if name in ('native_cold_serial','native_cold_early','native_hot',
                                'optimized_cold_serial','optimized_cold_early','optimized_hot'):
                        torch.testing.assert_close(captured_output,hidden*8,rtol=0,atol=0)
                    if name.startswith('native_cold'):
                        check_weight_slots(pool,split.plan.experts_to_copy)
                    if name.startswith('optimized_cold'):
                        check_weight_slots(pool,mat.plan.experts_to_copy)
                    if name in ('exchange_prefix','optimized_full_planner') or name.startswith('optimized_cold') or name=='optimized_hot':
                        assert torch.equal(fast.owner,a),'graph optimized owner differs reference'
                        assert torch.equal(fast.stats,old.stats),'graph exchange stats differ reference'
                        mat.check_errors()
                    if name=='exchange_sort':assert torch.equal(old.owner,a),'graph sort owner differs reference'
                    if name=='native_planner':assert torch.equal(decode(),seed_cpu),'graph native owner differs CPU'
                row['graph_capture_passed']=True
            names=list(functions)
            def prepare_hot(name):
                # These variants share physical copy slots. A previously run
                # variant may have installed a different expert in a slot.
                # Establish the advertised warm state explicitly, untimed.
                if name=='native_hot':
                    p,_,r=native()
                elif name=='optimized_hot':
                    p,_,r,_=opt_plan(False)
                else:return None
                w=split.prefetch(p.experts_to_copy,ready=r,**pool.kwargs());w.synchronize()
                check_weight_slots(pool,p.experts_to_copy)
                return hashlib.sha256(p.experts_to_copy.cpu().numpy().tobytes()).hexdigest()
            for rep in range(args.samples):
                order=names[rep%len(names):]+names[:rep%len(names)]
                for name in order:
                    hot_hash=prepare_hot(name)
                    callback=graphs[name].replay if args.mode=='graph' else functions[name]
                    measured=gather_object(timed(callback))
                    row['measurements'][name].append(dict(per_rank=measured,
                        max_cuda_ms=max(x['cuda_ms'] for x in measured),max_host_ms=max(x['host_ms'] for x in measured),
                        hot_prepared_map_sha256=hot_hash))
                    if hot_hash is not None:
                        p=split.plan if name=='native_hot' else mat.plan
                        assert hashlib.sha256(p.experts_to_copy.cpu().numpy().tobytes()).hexdigest()==hot_hash
                        check_weight_slots(pool,p.experts_to_copy)
                save()
            mat.check_errors();assert torch.equal(res.support,support_cpu)
            # Destroy captured NCCL graph references before freeing buffers or
            # destroying the process group. Retained graph callbacks can keep
            # NCCL graph user objects alive during shutdown indefinitely.
            if args.mode=='graph':
                callback=None;captured_output=None
                for captured_graph in graphs.values():captured_graph.reset()
                graphs.clear();graph=None;captured_graph=None
                torch.cuda.synchronize();dist.barrier()
            if args.timeline and not args.components_only and layer in (0,7):
                from timeline_overlap_graph import instrument_pipeline
                row['timeline_diagnostic']=gather_object(instrument_pipeline(
                    native=native,decode=decode,residency=res,materializer=mat,
                    exchange=fast,routes=routes,hidden=hidden,pool=pool,
                    buffer=buffer,split=split,optimized=True,early=True,execution=args.mode,
                    trace_path=args.output.parent/f'{args.output.stem}_timeline_l{layer:02}_rank{rank}.json'))
            row['post_timing_checked']=True;save()
            del functions,fast,old,mat,sort_mat,reuse_mat,other_mat,res,routes
        record['status']='complete';save()
    except Exception as exc:
        record['status']='failed';record['failures'].append(dict(rank=rank,error=repr(exc),traceback=traceback.format_exc()));save();raise
    finally:
        if split is not None:split.close()
        # Keep mapped weight owners alive until all consumers have drained.
        if pool is not None and hasattr(pool,'close'):pool.close()
        if buffer is not None:buffer.destroy()
        dist.destroy_process_group()

if __name__=='__main__':main()
