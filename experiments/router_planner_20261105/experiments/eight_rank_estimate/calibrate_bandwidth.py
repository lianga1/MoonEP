"""Two physical H200 calibration ONLY; never folds or optimizes the real trace.

Native MoonEP BF16 prefetch, FP32 gradient return, and BF16 dispatch/combine.
Report payload bytes per sending/receiving endpoint, not sum(TX+RX) bandwidth.
The eight-rank forecast consumes these rates with its ORIGINAL traffic matrices.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, statistics, sys, traceback
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/two_gpu_overlap'),str(ROOT/'inputs/MoonEP')]
from benchmark import gather_object,timed
from moonep_common import create_buffer,compile_native,allocate_weights,check_weight_slots
from moonep_split import SplitPhaseMoonEP
from training_grad import allocate_gradients

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--samples',type=int,default=10)
    ap.add_argument('--tokens',default='16768,65536')
    ap.add_argument('--hidden',type=int,default=7168)
    ap.add_argument('--intermediate',type=int,default=2048)
    args=ap.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=15))
    assert dist.get_world_size()==2
    if args.output.exists():raise FileExistsError('Keep previous attempts')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rec=dict(status='running',metadata=dict(physical_gpus=2,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
        H=args.hidden,I=args.intermediate,weight_bytes=3*args.hidden*args.intermediate*2,
        gradient_bytes=3*args.hidden*args.intermediate*4,num_sms=32,prefetch_sms=32,
        scope='bandwidth calibration, synthetic balanced routes; not a two-rank routing-quality result',
        timing='CUDA Graph, max rank elapsed/sample; initialization and validation excluded',
        bandwidth='max per-endpoint payload bytes / max rank time; decimal GB/s, one direction, no TX+RX double count',
        extrapolation='8-way NVSwitch contention and native R8 synchronization are not measured',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),ROOT/'experiments/two_gpu_overlap/training_grad.py',
                  ROOT/'experiments/two_gpu_overlap/moonep_split.py']}),cases=[],failures=[])
    def save():
        if rank==0:
            args.output.write_text(json.dumps(rec,indent=2)+'\n')
            print(json.dumps(dict(status=rec['status'],cases=len(rec['cases']))),flush=True)
    def measure(name,fn,txbytes,rxbytes,prepare=None,check=None,extra=None):
        if prepare:prepare()
        fn();torch.cuda.synchronize()
        if check:check()
        for _ in range(3):
            if prepare:prepare()
            dist.barrier();torch.cuda.synchronize();fn();torch.cuda.synchronize()
        if prepare:prepare()
        dist.barrier();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        # Capture records a future event, not a completed launch. All previous
        # work was synchronized above; reset must not wait on that unlaunched
        # capture event before the first replay.
        if grads is not None:grads._done=None
        row=dict(name=name,tx_bytes_by_rank=txbytes,rx_bytes_by_rank=rxbytes,
                 extra=extra or {},raw=[],correctness_checked=check is not None)
        rec['cases'].append(row)
        for sample in range(args.samples):
            if prepare:prepare()
            per=gather_object(timed(graph.replay))
            row['raw'].append(dict(sample=sample,per_rank=per,
                max_cuda_ms=max(x['cuda_ms'] for x in per),max_host_ms=max(x['host_ms'] for x in per)))
        if check:check()
        ms=statistics.median(x['max_cuda_ms'] for x in row['raw'])
        row['median_ms']=ms
        row['effective_endpoint_GBs']=max(txbytes+rxbytes)/(ms*1e6)
        torch.cuda.synchronize()
        # Graph-owned event handles cease to be usable after graph.reset().
        # There is no outstanding work here; discard cached handles first.
        if grads is not None:grads._done=None
        if split is not None:split._weight_done=None
        graph.reset();save()
    buffer=split=weights=grads=None
    save()
    try:
        for si,S in enumerate(map(int,args.tokens.split(','))):
            buffer=create_buffer(S,args.hidden,E=16,B=8,num_sms=32)
            compile_native(buffer,prefetch_sms=32,weight_shapes=[
                (16,2*args.intermediate,args.hidden,8,torch.bfloat16),
                (16,args.hidden,args.intermediate,8,torch.bfloat16)])
            split=SplitPhaseMoonEP(buffer,prefetch_sms=32)
            # Both endpoints choose four home experts and four remote experts.
            # Uniform expert totals are already balanced, no replicas needed.
            topk=torch.tensor([0,1,2,3,8,9,10,11],device='cuda',dtype=torch.int32).expand(S,8).contiguous()
            tpe=torch.bincount(topk.flatten().long(),minlength=16).int()
            p,cu,ready=split.plan_native(topk,tpe);torch.cuda.synchronize()
            assert int((p.experts_to_copy>=0).sum())==0
            x=(torch.arange(S,device='cuda')%11+rank*13).to(torch.bfloat16)[:,None].expand(S,args.hidden).contiguous()
            router=torch.ones((S,8),device='cuda',dtype=torch.float32)
            ticket=split.dispatch_fresh(x,router,plan=p,cu_seqlens=cu,ready=ready)
            torch.cuda.current_stream().wait_event(ticket.done)
            y=buffer.combine(plan=p,hidden_nvsh=ticket.hidden,zero_copy=True)[0]
            torch.cuda.synchronize();torch.testing.assert_close(y,x*8,rtol=0,atol=0)
            def dispatch():
                out=buffer.dispatch(x,router,plan=p,async_finish=True,zero_copy=True)
                torch.cuda.current_stream().wait_event(out[-1])
            def combine():buffer.combine(plan=p,hidden_nvsh=ticket.hidden,zero_copy=True)
            def token_check():
                dispatch();out=buffer.combine(plan=p,hidden_nvsh=ticket.hidden,zero_copy=True)[0]
                torch.cuda.synchronize();torch.testing.assert_close(out,x*8,rtol=0,atol=0)
                return True
            payload=S*args.hidden*2
            measure(f'dispatch_S{S}',dispatch,[payload]*2,[payload]*2,check=token_check,
                    extra=dict(remote_destinations_per_rank=S,router_bytes_per_rank=S*8*4,
                               dedup='existing map; build excluded',activation_bytes_only_in_rate=True))
            measure(f'combine_S{S}',combine,[payload]*2,[payload]*2,check=token_check,
                    extra=dict(remote_destinations_per_rank=S,router_return=False))
            if si==0:
                weights=allocate_weights(buffer,args.hidden,args.intermediate);weights.fill_sources()
                grads=allocate_gradients(buffer,args.hidden,args.intermediate);grads.precompile()
                epoch=[0]
                def reset():
                    epoch[0]+=1;grads.reset(epoch=epoch[0])
                for n0,n1 in [(0,1),(1,1),(4,4),(8,8),(1,8),(8,1)]:
                    mapping=torch.full((2,8),-1,device='cuda',dtype=torch.int32)
                    mapping[0,:n0]=torch.arange(8,8+n0,device='cuda',dtype=torch.int32)
                    mapping[1,:n1]=torch.arange(n1,device='cuda',dtype=torch.int32)
                    def prefetch():
                        event=split.prefetch(mapping,ready=torch.cuda.current_stream().record_event(),**weights.kwargs())
                        torch.cuda.current_stream().wait_event(event)
                    def reduce():torch.cuda.current_stream().wait_event(grads.launch(mapping,publish=True))
                    b=weights.bytes_per_expert
                    measure(f'weight_{n0}_{n1}',prefetch,[n1*b,n0*b],[n0*b,n1*b],
                            check=lambda:check_weight_slots(weights,mapping),extra=dict(copies_by_destination=[n0,n1]))
                    b=grads.bytes_per_expert
                    measure(f'gradient_{n0}_{n1}',reduce,[n0*b,n1*b],[n1*b,n0*b],prepare=reset,
                            check=lambda:grads.check(mapping),extra=dict(copies_by_source=[n0,n1],input_ready_barrier_included=True))
                grads.close();grads=None;weights.close();weights=None
            split.close();split=None
            del p,cu,ready,ticket,x,router,topk,tpe,y
            torch.cuda.synchronize();dist.barrier();buffer.destroy();buffer=None
            torch.cuda.empty_cache()
        rec['status']='complete';save()
    except Exception:
        rec['status']='failed';rec['failures'].append(traceback.format_exc());save();raise
    finally:
        if grads is not None:grads.close()
        if weights is not None:weights.close()
        if split is not None:split.close()
        if buffer is not None:buffer.destroy()
        dist.destroy_process_group()

if __name__=='__main__':main()
