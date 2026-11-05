"""Untimed diagnostic of early expert copy and token communication overlap.

CUDA events delimit stream phases. Optional profiler export is required to
establish kernel-execution overlap: intersecting stream ranges alone also
include launch gaps/barriers and do not measure network byte occupancy.
"""
from contextlib import nullcontext
from pathlib import Path
import json,re
import torch
import torch.distributed as dist


def _intersection(a,b):
    return max(0.0,min(a[1],b[1])-max(a[0],b[0]))


def _merge(intervals):
    out=[]
    for start,end in sorted(intervals):
        if out and start<=out[-1][1]:out[-1][1]=max(out[-1][1],end)
        else:out.append([start,end])
    return out


def analyze_kernel_overlap(path):
    data=json.loads(Path(path).read_text());kernels=[]
    for e in data.get('traceEvents',[]):
        if e.get('ph')=='X' and 'kernel' in str(e.get('cat','')).lower() and 'dur' in e:
            kernels.append(e)
    weights=[e for e in kernels if re.search('prefetch',e['name'],re.I)]
    tokens=[e for e in kernels if re.search('dispatch',e['name'],re.I) and not re.search('epilogue',e['name'],re.I)]
    w=_merge([(e['ts'],e['ts']+e['dur']) for e in weights]);t=_merge([(e['ts'],e['ts']+e['dur']) for e in tokens])
    overlap=sum(_intersection(a,b) for a in w for b in t)/1000
    return dict(classification='case-insensitive prefetch vs dispatch excluding epilogue; inspect names',
                kernel_classification_complete=bool(weights and tokens),
                prefetch_kernel_names=sorted({e['name'] for e in weights}),
                dispatch_kernel_names=sorted({e['name'] for e in tokens}),
                all_kernel_names=sorted({e['name'] for e in kernels}),
                measured_kernel_overlap_ms=overlap if weights and tokens else None,
                note='Kernel execution overlap does not establish simultaneous network byte throughput.')


def instrument_pipeline(*,native,decode,residency,materializer,exchange,
                        routes,hidden,pool,buffer,split,optimized=True,early=True,
                        trace_path=None):
    """Run one cold pipeline with GPU timestamps; never mix into performance samples.

    native()->(plan,cu,event), decode()->local owner. All objects must be warmed
    and all first-use compilation complete on both ranks before this function.
    """
    torch.cuda.synchronize();dist.barrier()
    stamps={}
    def mark(name,stream=None):
        event=torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream() if stream is None else stream)
        stamps[name]=event
        return event
    def prefetch(ids,ready):
        producer=torch.cuda.current_stream().record_event()
        split.expert_stream.wait_event(ready);split.expert_stream.wait_event(producer)
        mark('weight_stream_start',split.expert_stream)
        done=split.prefetch(ids,ready=ready,**pool.kwargs())
        mark('weight_stream_end',split.expert_stream)
        return done
    profiler=(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False)
              if trace_path is not None else nullcontext())
    with profiler as prof:
        mark('begin')
        with torch.profiler.record_function('diagnostic_native_seed'):
            plan,cu,ready=native()
        mark('native_plan_ready')
        if optimized:
            with torch.profiler.record_function('diagnostic_residency'):
                owner=decode();support=residency.run(plan.experts_to_copy,owner,local_routes=routes)
                ids,expert_ready=materializer.prepare_support(support)
            mark('expert_plan_ready')
            weight_done=prefetch(ids,expert_ready) if early else None
            with torch.profiler.record_function('diagnostic_exchange_and_output'):
                final=exchange.run(owner,support=support)
                plan,cu,ready=materializer.run(final)
        else:
            mark('expert_plan_ready')
            weight_done=prefetch(plan.experts_to_copy,ready) if early else None
        mark('token_plan_ready')
        producer=torch.cuda.current_stream().record_event()
        split.token_stream.wait_event(ready);split.token_stream.wait_event(producer)
        mark('token_stream_start',split.token_stream)
        with torch.profiler.record_function('diagnostic_token_dispatch'):
            out=split.dispatch_fresh(hidden,None,plan=plan,cu_seqlens=cu,ready=ready)
        mark('token_stream_end',split.token_stream)
        if not early:weight_done=prefetch(plan.experts_to_copy,out.done)
        torch.cuda.current_stream().wait_event(out.done)
        torch.cuda.current_stream().wait_event(weight_done)
        mark('both_inputs_ready')
        with torch.profiler.record_function('diagnostic_combine'):
            result=buffer.combine(plan=plan,hidden_nvsh=out.hidden,zero_copy=True)[0]
        mark('combine_end')
        stamps['combine_end'].synchronize()
    base=stamps['begin'];times={k:base.elapsed_time(v) for k,v in stamps.items()}
    wr=[times['weight_stream_start'],times['weight_stream_end']]
    tr=[times['token_stream_start'],times['token_stream_end']]
    report=dict(rank=dist.get_rank(),optimized=optimized,early=early,
        scope='instrumented cold identity-expert communication segment; excluded from benchmark samples',
        gpu_event_ms=times,weight_token_stream_interval_overlap_ms=_intersection(wr,tr),
        expert_copy_lead_over_token_plan_ms=times['token_plan_ready']-times['weight_stream_start'],
        expert_copy_exposed_after_dispatch_ms=max(0.0,wr[1]-tr[1]),
        caution='Stream intervals include kernel-launch gaps and rank barriers; profiler kernel intersection is the stronger overlap evidence.')
    torch.testing.assert_close(result,hidden*int(buffer._require_ctx()['K']),rtol=0,atol=0)
    if trace_path is not None:
        path=Path(trace_path);path.parent.mkdir(parents=True,exist_ok=True)
        prof.export_chrome_trace(str(path));report['profiler']=analyze_kernel_overlap(path)
        report['trace_path']=str(path)
    return report
