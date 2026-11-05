"""Real router replay -> runtime GPU residency/exchange -> real DeepEP v1.

Two physical ranks, prepared MoonEP CPU-reference seed, source shards 0..3/4..7.
This is backend integration correctness, NOT native planner or model timing.
Hidden vectors are deterministic synthetic payloads with unique token identities.

  PYTHONPATH=<root>/experiments/two_gpu_overlap/vendor/DeepEP_v1 \
    torchrun --standalone --nproc_per_node=2 deepep_trace_smoke.py \
    --layers 0,7 --hidden 7168 --output <out>/deepep_trace.json

Optional --weights moonep launches real MoonEP one-sided weight prefetch as soon
as the GPU expert map is ready, before the subsequent full-token exchange. That
path also needs the existing MoonEP extension and its peer-mapped WeightPool.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist

from deepep_adapter import validate_plan
from deepep_v1_adapter import DeepEPV1PlannedAdapter, UPSTREAM_COMMIT, build_provenance
from exchange_cuda import TwoRankExchange
from residency_cuda import TwoRankResidency

ROOT = Path(__file__).resolve().parents[2]


def hidden_pattern(global_token_ids, hidden):
    """Base-16 token identity in the first five columns, then salted repeats.

    Five digits identify up to 2**20 distinct tokens. Values are integers 1..16;
    multiplying by any local expert count 0..8 and adding the two rank outputs
    is exact in BF16. Thus a zero-tolerance K*x check is meaningful here.
    """
    columns=torch.arange(80,device=global_token_ids.device,dtype=torch.int64)
    shifts=4*(columns%5)
    digits=(global_token_ids.to(torch.int64)[:,None]>>shifts[None,:])&15
    base=((digits+columns[None,:]//5)%16+1).to(torch.bfloat16)
    return base.index_select(1,torch.arange(hidden,device=base.device)%80).contiguous()


def gather_scalar(value):
    out=[None]*dist.get_world_size()
    dist.all_gather_object(out,value)
    return out


def audit_received(recv_x,recv_idx,physical,rank,local_experts):
    """Check exact source token AND its expert slot IDs, including permutations."""
    world=dist.get_world_size()
    global_physical=torch.empty((world*physical.shape[0],physical.shape[1]),
                                 device=physical.device,dtype=physical.dtype)
    dist.all_gather_into_tensor(global_physical,physical)
    valid_rows=(recv_idx>=0).any(dim=1)
    values=recv_x[valid_rows]
    idx=recv_idx[valid_rows]
    base16=torch.tensor([1,16,256,4096,65536],dtype=torch.int64,device=recv_x.device)
    global_ids=((values[:,:5].to(torch.int64)-1)*base16[None,:]).sum(dim=1)
    assert bool(((global_ids>=0)&(global_ids<global_physical.shape[0])).all()), 'invalid source token identity'
    assert global_ids.unique().numel()==global_ids.numel(), 'duplicate source token at destination'
    expected_x=hidden_pattern(global_ids,recv_x.shape[1])
    assert torch.equal(values,expected_x), 'received hidden payload differs from unique source pattern'
    expected=global_physical[global_ids]
    expected=torch.where(expected//local_experts==rank,expected%local_experts,-1)
    assert torch.equal(idx,expected), 'received expert slots differ from planner per-token choices'
    actual_counts=torch.bincount(idx[idx>=0].long(),minlength=local_experts)
    all_counts=torch.bincount(global_physical.flatten().long(),minlength=world*local_experts)
    assert torch.equal(actual_counts,all_counts[rank*local_experts:(rank+1)*local_experts]), 'expert task count mismatch'
    return dict(exact_source_token_and_payload=True,exact_per_token_expert_slots=True,
                unique_receive_token_per_rank=True,expert_counts_exact=True,
                received_valid_tokens=int(valid_rows.sum().item()),
                received_expert_tasks=int(actual_counts.sum().item()),
                receive_capacity=int(recv_x.shape[0]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--layers',default='0,7')
    p.add_argument('--hidden',type=int,default=7168)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--num-sms',type=int,default=16)
    p.add_argument('--weights',choices=('none','moonep'),default='none')
    p.add_argument('--intermediate',type=int,default=2048)
    p.add_argument('--prefetch-sms',type=int,default=8)
    p.add_argument('--input-dir',type=Path,default=ROOT/'outputs/two_gpu_overlap/inputs')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.hidden<256 or args.hidden%256:
        p.error('BF16 hidden must be divisible by 256; H128 has a retained combine failure; tested target H7168')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=10))
    rank=dist.get_rank()
    assert dist.get_world_size()==2
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',rank=rank,metadata=dict(
        physical_ranks=2,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),hidden=args.hidden,
        token_backend='DeepEP v1.2.1 intranode',reference_commit=UPSTREAM_COMMIT,
        expert_backend='MoonEP one-sided NVLink prefetch' if args.weights=='moonep' else 'not exercised in this probe',
        source_mapping='original trace shards 0..3 -> physical rank 0; 4..7 -> physical rank 1',
        scope='real router backend-adapter correctness; prepared reference seed; no native planner timing; no GEMM',
        hidden_payload='synthetic BF16 unique per-token base16 pattern',
        online_inputs='routes and owners sampled on GPU each run with one joint collective; d IDs fixed',
        sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*')
                 if f.suffix in ('.py','.cu')}),layers=[],failures=[])
    def save():
        output.write_text(json.dumps(record,indent=2)+'\n')
    adapter=None;weight_buffer=None;pool=None;split=None
    try:
        record['metadata']['build_provenance']=build_provenance()
        save()
        for layer in map(int,args.layers.split(',')):
            f=args.input_dir/f'layer_{layer:02}.npz'
            with np.load(f,allow_pickle=False) as data:
                S=data['topk'].shape[1]
                if S*2>=1<<20:
                    raise ValueError('hidden token-ID encoding capacity exceeded')
                routes=torch.tensor(data['topk'][rank],dtype=torch.int32,device='cuda')
                seed=torch.tensor(data['owner'][rank],dtype=torch.int32,device='cuda')
                expected_support=torch.tensor(data['support'],dtype=torch.bool,device='cuda')
                native_copies=torch.tensor(data['native_experts_to_copy'],dtype=torch.int32,device='cuda')
                sample_routes=torch.tensor(data['sample_routes'],dtype=torch.int32,device='cuda')
                sample_ids=torch.tensor(data['sample_ids'],dtype=torch.int64,device='cuda')
            if adapter is None:
                adapter=DeepEPV1PlannedAdapter(dist.group.WORLD,per_source=S,hidden=args.hidden,
                    topk=8,num_experts=352,extra_slots=8,num_sms=args.num_sms)
                if args.weights=='moonep':
                    sys.path.insert(0,str(ROOT/'inputs/MoonEP'))
                    from moonep_common import create_buffer,allocate_weights,check_weight_slots
                    from moonep_split import SplitPhaseMoonEP
                    from moonep import prefetch
                    # Small token capacity: this Buffer only owns weight mappings;
                    # DeepEP performs all token communication in this probe.
                    weight_buffer=create_buffer(128,args.hidden,num_sms=args.num_sms)
                    split=SplitPhaseMoonEP(weight_buffer,prefetch_sms=args.prefetch_sms)
                    for h,hp in ((2*args.intermediate,args.hidden),(args.hidden,args.intermediate)):
                        prefetch._get_compiled(352,h,hp,8,args.prefetch_sms,
                            torch.cuda.current_device(),torch.bfloat16)
                    dist.barrier()
                    pool=allocate_weights(weight_buffer,args.hidden,args.intermediate)
                    pool.fill_sources()
            if adapter.per_source!=S:
                raise ValueError('all selected layers must have equal source capacity')
            res=TwoRankResidency(sample_routes,sample_ids,local_tokens=S,rank=rank,slots=8)
            exchange=TwoRankExchange(routes,expected_support,rank=rank,
                                     rounds=args.rounds,variant='prefix')
            global_ids=rank*S+torch.arange(S,device='cuda',dtype=torch.int64)
            hidden=hidden_pattern(global_ids,args.hidden)
            router=torch.ones((S,8),device='cuda',dtype=torch.float32)
            row=dict(layer=layer,input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                     shape=[2,S,8],checks=[])
            record['layers'].append(row);save()
            for no_cpu_sync in (False,True):
                support=res.run(native_copies,seed,local_routes=routes)
                expert_map=res.experts_to_copy
                expert_ready=torch.cuda.current_stream().record_event()
                weights_done=(split.prefetch(expert_map,ready=expert_ready,**pool.kwargs())
                              if split is not None else None)
                # Weight prefetch is now free to run while exchange scans tokens.
                owner=exchange.run(seed,support=support)
                ticket=adapter.start_token_dispatch(hidden,routes,owner,expert_map,router,
                                                     no_cpu_sync=no_cpu_sync)
                recv_x,recv_idx,recv_router,counts,handle=ticket.wait()
                if weights_done is not None:
                    torch.cuda.current_stream().wait_event(weights_done)
                assert torch.equal(support,expected_support),'runtime GPU residency differs from prepared CPU reference'
                validate_plan(routes,owner,expert_map,352)
                load=torch.bincount(owner.flatten().long(),minlength=2)
                dist.all_reduce(load)
                assert load.tolist()==[S*8,S*8], 'load not conserved'
                received=audit_received(recv_x,recv_idx,ticket.physical,rank,184)
                valid=recv_idx>=0
                assert bool((recv_router[valid]==1).all()), 'route weight payload corrupted'
                count=valid.sum(dim=1)
                out=(recv_x.float()*count[:,None]).to(torch.bfloat16)
                out.masked_fill_(~valid.any(dim=1)[:,None],0)
                combined,_,done=adapter.combine(out,handle)
                done.current_stream_wait()
                expected=hidden*8
                torch.cuda.synchronize()
                torch.testing.assert_close(combined,expected,rtol=0,atol=0)
                weight_check=check_weight_slots(pool,expert_map) if pool is not None else None
                before=int((seed!=rank).any(dim=1).sum().item())
                after=int((owner!=rank).any(dim=1).sum().item())
                row['checks'].append(dict(no_cpu_sync=no_cpu_sync,
                    runtime_residency_cpu_equal=True,balanced_global_load=load.tolist(),
                    physical_slots_valid=True,identity_combine_exact=True,
                    max_abs_error=0.0,received=received,weight_check=weight_check,
                    remote_before_by_source=gather_scalar(before),
                    remote_after_by_source=gather_scalar(after),
                    expert_map=expert_map.cpu().tolist(),exchange_stats=exchange.stats.cpu().tolist()))
                save()
            del res,exchange,routes,seed,hidden,recv_x,recv_idx,recv_router,out,combined,expected,ticket
        record['status']='passed';save()
    except Exception as exc:
        record['status']='failed'
        record['failures'].append(dict(error=repr(exc),traceback=traceback.format_exc()))
        save()
        raise
    finally:
        if split is not None:split.close()
        if pool is not None:pool.close()
        if weight_buffer is not None:weight_buffer.destroy()
        if adapter is not None:adapter.destroy()
        dist.destroy_process_group()


if __name__=='__main__':
    main()
