"""Two physical GPU correctness smoke for native + split MoonEP integration.

Run from repository root with PYTHONPATH=inputs/MoonEP:$PYTHONPATH:
 torchrun --standalone --nproc_per_node=2 experiments/two_gpu_overlap/two_gpu_smoke.py --tokens 256 --hidden 128 --intermediate 1024
Use --tokens 0 --hidden 7168 --intermediate 2048 for the complete folded trace.
This is an audit, not a timing benchmark; its synthetic payload has known values.
"""
import argparse,datetime,hashlib,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from moonep_common import create_buffer,compile_native,allocate_weights,check_weight_slots
from moonep_split import SplitPhaseMoonEP
from moonep_materialize import CudaMoonEPMaterializer
from moonep_materialize_reference import reference_native_plan
from exchange_cuda import TwoRankExchange
from moonep_cpu import plan as cpu_plan,communication


def gather_numpy(tensor):
    items=[torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(items,tensor)
    return torch.stack(items).cpu().numpy()


def check_dispatch_combine(buffer,split,plan,cu,ready,hidden,weights,ref,rank):
    out=split.dispatch_fresh(hidden,weights,plan=plan,cu_seqlens=cu,ready=ready)
    torch.cuda.current_stream().wait_event(out.done)
    info=ref['src_info'][rank];valid=np.flatnonzero(info>=0);nvs=int(buffer._require_ctx()['NvS']);S=hidden.shape[0];K=weights.shape[1]
    src=info[valid]//nvs;off=info[valid]%nvs
    expected=(off//K)%11+src*13
    idx=torch.tensor(valid,device=hidden.device,dtype=torch.int64)
    values=torch.tensor(expected,device=hidden.device,dtype=torch.bfloat16)
    assert torch.all(out.hidden[idx]==values[:,None]).item(),'dispatched hidden provenance mismatch'
    assert torch.all(out.route_weights[idx]==1.0/K).item(),'dispatched route weight mismatch'
    # No expert GEMM: each of the K identical task outputs returns its input.
    # combine sums them; integer-valued BF16 payload keeps the expected result exact.
    restored,_,_=buffer.combine(plan=plan,hidden_nvsh=out.hidden,zero_copy=True)
    assert torch.equal(restored,hidden*K),'combine identity failed'
    return dict(valid_rows=len(valid),dispatch_hidden=True,dispatch_weights=True,combine_identity=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,default=0);p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--hidden',type=int,default=128);p.add_argument('--intermediate',type=int,default=1024)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/two_gpu_overlap/smoke_attempt01.json')
    p.add_argument('--num-sms',type=int,default=32);p.add_argument('--prefetch-sms',type=int,default=8)
    p.add_argument('--materializer',choices=('sort','reuse'),default='sort')
    args=p.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    assert dist.get_world_size()==2
    result={'status':'running','layer':args.layer,'ranks':2,'tokens_per_rank':args.tokens,'hidden':args.hidden,'intermediate':args.intermediate,'materializer':args.materializer,'failures':[]}
    buffer=split=pool=None
    try:
        path=ROOT/f'outputs/trace_audit/layer_{args.layer:02d}.npz'
        x=np.load(path,allow_pickle=False)['topk'].reshape(2,-1,8).copy()
        if args.tokens:x=x[:,:args.tokens].copy()
        R,S,K=x.shape;E=352;B=8
        result.update(tokens_per_rank=S,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trace_transform='contiguous eight-source rows folded into two sources; optional per-source prefix')
        cpu=cpu_plan(x,E=E,B=B,replica_target=0)
        assert np.all(cpu.physical_topk>=0),'native seed does not fit B slots'
        routes=torch.tensor(x[rank],device='cuda',dtype=torch.int32)
        tpe=torch.tensor(cpu.tpe[rank],device='cuda',dtype=torch.int32)
        buffer=create_buffer(S,args.hidden,K=K,E=E,B=B,num_sms=args.num_sms)
        c=buffer._require_ctx();split=SplitPhaseMoonEP(buffer,prefetch_sms=args.prefetch_sms)
        pool=allocate_weights(buffer,args.hidden,args.intermediate);pool.fill_sources()
        shapes=[(E,2*args.intermediate,args.hidden,B,torch.bfloat16),(E,args.hidden,args.intermediate,B,torch.bfloat16)]
        compile_native(buffer,prefetch_sms=args.prefetch_sms,weight_shapes=shapes)
        materializer_cls=CudaMoonEPMaterializer
        if args.materializer=='reuse':
            from moonep_materialize_reuse import CudaMoonEPMaterializerReuse
            materializer_cls=CudaMoonEPMaterializerReuse
        material=materializer_cls(buffer,routes)
        # Both ranks must finish the optional extension compilation before
        # either launches native planning's bounded inter-rank spin barrier.
        dist.barrier()
        print(f'rank {rank}: precompile complete',flush=True)
        native,native_cu,native_ready=split.plan_native(routes,tpe)
        torch.cuda.synchronize()
        native_owner=material.decode_native_owner(native)
        assert np.array_equal(native_owner.cpu().numpy(),cpu.owner[rank]),'native owner != CPU'
        assert np.array_equal(native.physical_topk.cpu().numpy().reshape(S,K),cpu.physical_topk[rank]),'native physical != CPU'
        assert np.array_equal(native.experts_to_copy.cpu().numpy(),cpu.experts_to_copy),'native replica IDs != CPU'
        # Build references using the actual native slot order; the native C2
        # row order is not the sorted-ID replica order used by our materializer.
        raw=native.dst.cpu().numpy();raw=np.where(raw>=0,raw,-raw-1).reshape(S,K)
        raw_all=gather_numpy(torch.tensor(raw,device='cuda',dtype=torch.int32))
        src_ref=np.full((R,int(c['NvS'])),-1,np.int32)
        for source in range(R):
            rr=raw_all[source].ravel();dest=rr//int(c['NvS']);off=rr%int(c['NvS'])
            src_ref[dest,off]=source*int(c['NvS'])+np.arange(S*K,dtype=np.int32)
        native_ref={'src_info':src_ref}
        hidden=((torch.arange(S,device='cuda',dtype=torch.int32)%11)+rank*13).to(torch.bfloat16)[:,None].expand(S,args.hidden).contiguous()
        weights=torch.full((S,K),1.0/K,dtype=torch.float32,device='cuda')
        native_checks=check_dispatch_combine(buffer,split,native,native_cu,native_ready,hidden,weights,native_ref,rank)
        native_weight=split.prefetch(native.experts_to_copy,ready=native_ready,**pool.kwargs());native_weight.synchronize()
        native_checks['weight_slots']=check_weight_slots(pool,native.experts_to_copy)
        # Give every rank legal spare replicas, then exercise the full exchange
        # and address builder. This is a deterministic audit support, not a claim
        # about the sampled residency algorithm's communication quality.
        support=(cpu.alloc.T>0)|(np.arange(E)[None,:]//(E//R)==np.arange(R)[:,None])
        for r in range(R):
            free=B-np.count_nonzero(support[r]&(np.arange(E)//(E//R)!=r))
            candidates=np.flatnonzero(~support[r]);scores=np.bincount(x[r].ravel(),minlength=E)
            order=np.lexsort((candidates,-scores[candidates]));support[r,candidates[order[:free]]]=True
        support_gpu=torch.tensor(support,device='cuda',dtype=torch.bool)
        exchange=TwoRankExchange(routes,support_gpu,rank=rank,rounds=4,variant='prefix')
        seed=torch.tensor(cpu.owner[rank],device='cuda',dtype=torch.int32)
        owner=exchange.run(seed);owners=gather_numpy(owner)
        assert np.array_equal(np.bincount(owners.ravel(),minlength=R),[S*K]*R)
        assert np.all(support[owners,x])
        ref=reference_native_plan(x,owners,support,B=B,padding=int(c['token_padding']),NvS=int(c['NvS']))
        etc,expert_ready=material.prepare_support(support_gpu)
        weight_done=split.prefetch(etc,ready=expert_ready,**pool.kwargs())
        planned,cu,ready=material.run(owner);torch.cuda.synchronize();material.check_errors()
        for name,actual,wanted in (
          ('dst',planned.dst,ref['dst'][rank].ravel()),('physical_topk',planned.physical_topk,ref['physical_topk'][rank].ravel()),
          ('experts_to_copy',planned.experts_to_copy,ref['experts_to_copy']),('cu_seqlens',cu,ref['cu_seqlens'][rank]),
          ('zero_fill_ranges',planned.zero_fill_ranges,ref['zero_fill_ranges'][rank]),('remote_stats',planned.remote_stats,ref['remote_stats'][rank])):
            assert np.array_equal(actual.cpu().numpy(),wanted),name+' metadata mismatch'
        src_offset=rank*int(c['meta_chunk_padded'])+int(c['SRC_INFO_OFF'])
        actual_src=c['meta_buf'][src_offset:src_offset+int(c['NvS'])].cpu().numpy()
        assert np.array_equal(actual_src,ref['src_info'][rank]),'source provenance mismatch'
        optimized_checks=check_dispatch_combine(buffer,split,planned,cu,ready,hidden,weights,ref,rank)
        weight_done.synchronize();optimized_checks['weight_slots']=check_weight_slots(pool,etc)
        checks=[None]*R;dist.all_gather_object(checks,dict(rank=rank,native=native_checks,optimized=optimized_checks))
        result.update(status='complete',checks=checks,native_matches_cpu=True,all_materialized_fields_match=True,
                      seed_communication=communication(x,cpu.owner,E),optimized_communication=communication(x,owners,E),
                      weight_bytes_per_expert=pool.bytes_per_expert,physical_devices=gather_numpy(torch.tensor([torch.cuda.get_device_properties(local).total_memory],device='cuda',dtype=torch.int64)).tolist())
    except BaseException as exc:
        result['status']='failed';result['failures'].append(dict(rank=rank,error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if rank==0:
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
        if split is not None:split.close()
        if pool is not None:pool.close()
        if buffer is not None:buffer.destroy()
        if dist.is_initialized():dist.destroy_process_group()
    if rank==0:print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
