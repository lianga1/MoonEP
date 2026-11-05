"""Original-eight-logical-rank GPU components on one/two independent devices.

Each source scan processes exactly 16,768 original tokens with all eight possible
execution destinations. No two-rank routing, division-by-eight latency estimate,
or real eight-rank collective is used. A sequential one-GPU execution validates
live count/quota dependencies before timing source computations with explicitly
precomputed round counts. The latter are CONDITIONAL components, not a complete
distributed planner. torchrun may launch two independent physical-GPU replicas;
this program deliberately does not initialize a distributed process group.
"""
from __future__ import annotations
import argparse,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/h100_planner'))
from bundle_lane8 import BucketCudaPlanner
from baseline_virtual import VirtualMoonEPPlanner
from exchange_prefix import EightRankPrefix,BUCKETS


def measure(fn, *, samples, repeats, warmup, mode):
    """CUDA events and host observation; compilation/allocation excluded."""
    for _ in range(warmup):fn()
    torch.cuda.synchronize()
    graph=None
    try:
        if mode=='graph':
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            graph.replay();torch.cuda.synchronize()
        call=graph.replay if graph is not None else fn
        rows=[]
        for index in range(samples):
            torch.cuda.synchronize()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();tick=time.perf_counter()
            for _ in range(repeats):call()
            end.record();end.synchronize()
            rows.append(dict(sample=index,cuda_ms=start.elapsed_time(end)/repeats,
                             host_ms=(time.perf_counter()-tick)*1000/repeats))
        return dict(execution_mode=mode,replays_or_calls_per_sample=repeats,samples=rows,
                    median_cuda_ms=statistics.median(r['cuda_ms'] for r in rows),
                    median_host_ms=statistics.median(r['host_ms'] for r in rows))
    finally:
        if graph is not None:
            torch.cuda.synchronize();graph.reset()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=ROOT/'outputs/eight_rank_estimate/kernel_inputs')
    ap.add_argument('--layers',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--sources',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',type=int,default=int(os.environ.get('LOCAL_RANK','0')))
    ap.add_argument('--samples',type=int,default=10);ap.add_argument('--repeats',type=int,default=20)
    ap.add_argument('--warmup',type=int,default=3);ap.add_argument('--mode',choices=('eager','graph'),default='graph')
    ap.add_argument('--include-virtual-seed',action='store_true')
    ap.add_argument('--require-residency',action='store_true')
    ap.add_argument('--skip-residency',action='store_true')
    args=ap.parse_args();sources=list(map(int,args.sources.split(',')))
    if len(set(sources))!=len(sources) or any(s not in range(8) for s in sources):raise ValueError('unique logical source IDs 0..7 required')
    if min(args.samples,args.repeats,args.warmup)<1:raise ValueError('positive sample/repeat/warmup counts required')
    if args.require_residency and args.skip_residency:raise ValueError('require/skip residency are mutually exclusive')
    process_rank=int(os.environ.get('RANK','0'));physical_world=int(os.environ.get('WORLD_SIZE','1'))
    output=args.output
    if physical_world>1:output=output.with_name(output.stem+f'_physical{process_rank}'+output.suffix)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write('{"status":"initializing"}\n')
    torch.cuda.set_device(args.device);torch.set_num_threads(1)
    props=torch.cuda.get_device_properties(args.device)
    record=dict(status='running',metadata=dict(
        logical_ranks=8,global_tokens=134144,tokens_per_source=16768,K=8,E=352,B=8,d=256,g=2,L=4,sample_seed=20261026,
        physical_process_rank=process_rank,physical_process_count=physical_world,device=args.device,
        gpu=torch.cuda.get_device_name(args.device),uuid=str(getattr(props,'uuid','unavailable')),
        torch=torch.__version__,cuda=torch.version.cuda,execution_mode=args.mode,
        source_mapping='original source IDs preserved exactly; physical processes independently replay logical inputs',
        scope='source-shard GPU component timings plus optional single-device global selector; no eight-rank distributed timing',
        exclusions=['real eight-rank sample gather','real eight-rank count collective','expert-plan broadcast',
                    'native MoonEP eight-device planner','physical address publication','weights/tokens/gradients/GEMM'],
        no_divide_by_eight=True,source_scans_are_actual_16768_token_work=True,
        native_residency='rt0 balanced seed from original eight-rank saved reference',
        source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (Path(__file__).parent,ROOT/'experiments/h100_planner')
            for p in directory.glob('*') if p.suffix in ('.py','.cu')}),layers=[],failures=[])
    def save():output.write_text(json.dumps(record,indent=2)+'\n')
    def timed(fn):return measure(fn,samples=args.samples,repeats=args.repeats,warmup=args.warmup,mode=args.mode)
    save()
    try:
        for layer in map(int,args.layers.split(',')):
            path=args.input_dir/f'layer_{layer:02}.npz';data=np.load(path,allow_pickle=False)
            assert data['topk'].shape==(8,16768,8)
            def gpu(array,dtype=torch.int32):return torch.tensor(array,dtype=dtype,device='cuda')
            routes=gpu(data['topk']);seed=gpu(data['seed']);support=gpu(data['support'],torch.bool)
            expected=gpu(data['owner_prefixes']);expected_stats=gpu(data['stats'],torch.int64)
            native_etc=gpu(data['native_experts_to_copy']);sample_ids=gpu(data['sample_ids'],torch.int64)
            row=dict(layer=layer,input_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     shape=list(data['topk'].shape),correctness={},global_components={},source_components=[])
            record['layers'].append(row);save()
            module=Path(__file__).with_name('residency8.py')
            if module.exists() and not args.skip_residency:
                from residency8 import Residency8
                residency=Residency8(routes.reshape(-1,8),sample_ids,per_source=16768,slots=8)
                result=residency.run(seed.reshape(-1,8),native_etc)
                assert torch.equal(result,support),'R8 GPU residency differs original CPU support'
                row['correctness']['global_residency_support_exact']=True
                row['global_components']['residency']=dict(
                    scope='complete R8 GPU selection with original all-source routes/owners already resident on this GPU; runtime sample-index gather included',
                    excluded='physical eight-rank sample collection and support publication',
                    timing=timed(lambda:residency.run(seed.reshape(-1,8),native_etc)))
                assert torch.equal(residency.support,support)
            else:
                if args.require_residency:raise FileNotFoundError(module)
                row['global_components']['residency']=dict(status='not_implemented_or_not_available',
                    timing=None,warning='R2 fused selector timing is not an R8 estimate; fixed support exchange is not a full planner')
            if args.include_virtual_seed:
                virtual=VirtualMoonEPPlanner(routes.reshape(-1,8))
                got=virtual.run_seed();assert torch.equal(got.reshape_as(seed),seed)
                row['global_components']['virtual_seed']=dict(
                    scope='whole global NK semantic rt0 CUDA port on one device, NOT native MoonEP or per-source timing',
                    excluded='local histogram preparation and physical eight-rank communication',
                    timing=timed(virtual.run_seed))
                assert torch.equal(virtual.owner.reshape_as(seed),seed)
            prefix=[EightRankPrefix(routes[s],support,source=s,rounds=4) for s in range(8)]
            sort=[BucketCudaPlanner(routes[s],support,per_source=16768,rounds=4,
                                    global_token_offset=s*16768,coordinated=False) for s in range(8)]
            counts=torch.empty((4,8,BUCKETS),dtype=torch.int32,device='cuda')
            for s in range(8):prefix[s].prepare_support();prefix[s].reset(seed[s]);sort[s].reset(seed[s])
            # Live coupled correctness: all source proposals precede global count
            # assembly; only those current counts are allowed to determine quota.
            for step in range(4):
                for s in range(8):
                    prefix[s].propose(step);sort[s].propose(step)
                    assert torch.equal(prefix[s].keys,(sort[s].keys>>sort[s].token_bits).to(torch.int32)),('proposal bucket',s,step)
                    assert torch.equal(prefix[s].moves,sort[s].move_masks),('proposal subset',s,step)
                    assert torch.equal(prefix[s].counts,sort[s].counts),('proposal count',s,step)
                    counts[step,s].copy_(prefix[s].counts)
                for s in range(8):
                    prefix[s].commit(step,counts[step]);sort[s].commit(step,counts[step],s,8)
                    assert torch.equal(prefix[s].owner,expected[step+1,s]),('CPU prefix owner',s,step)
                    assert torch.equal(sort[s].owner,expected[step+1,s]),('CPU sort owner',s,step)
                    assert torch.equal(prefix[s].stats[step],expected_stats[step]),('CPU prefix global stats',s,step)
                    assert torch.equal(sort[s].stats[step],expected_stats[step]),('CPU sort global stats',s,step)
            row['correctness'].update(all_32_source_round_owners_cpu_exact=True,
                prefix_sort_proposals_masks_counts_exact=True,global_quotas_use_live_all8_source_counts=True)
            row['count_dependency']=dict(buckets_per_source=BUCKETS,bytes_per_source=BUCKETS*4,
                bytes_global_table=8*BUCKETS*4,rounds=4,
                measured_full_eight_rank_collective=False,
                note='The saved tables below were generated by live coupled validation; timed source replays condition on their availability.')
            for source in sources:
                fast,old=prefix[source],sort[source]
                sr=dict(source=source,tokens_scanned=16768,possible_execution_ranks=8,rounds=[])
                row['source_components'].append(sr)
                sr['prepare_residency_cache']=timed(fast.prepare_support)
                def run_fast():return fast.run_with_external_counts(seed[source],counts,prepare=True)
                def run_sort():
                    old.ext.set_support(support,old.resident);old.reset(seed[source])
                    for step in range(4):old.propose(step);old.commit(step,counts[step],source,8)
                    return old.owner
                sr['prefix_L4_conditional']=timed(run_fast)
                assert torch.equal(fast.owner,expected[4,source])
                sr['sort_L4_conditional']=timed(run_sort)
                assert torch.equal(old.owner,expected[4,source])
                sr['L4_scope']='includes support/cache preparation, seed restore, four local scans, GPU fold/quota, commit; fixed round counts available; excludes physical collective and waiting'
                for step in range(4):
                    fast.reset(expected[step,source]);fast.prepare_support();old.reset(expected[step,source])
                    rr=dict(round=step)
                    rr['prefix_proposal_and_local_counts']=timed(lambda:fast.propose(step))
                    rr['sort_proposal_and_local_counts']=timed(lambda:old.propose(step))
                    rr['fold_all8_counts_and_quota']=timed(lambda:fast.quota(step,counts[step]))
                    fast.propose(step);fast.quota(step,counts[step])
                    rr['prefix_commit_only']=timed(fast.apply)
                    rr['commit_only_scope']='fixed proposal/ordinal/quota; repeated writes are idempotent (clear old-owner bits, set destination bits, assign identical destination); does not run another planning round'
                    assert torch.equal(fast.owner,expected[step+1,source])
                    sr['rounds'].append(rr)
                save()
            row['post_timing_correctness']=True;save()
            del prefix,sort,routes,seed,support,expected,expected_stats,counts
        record['status']='complete';save()
    except BaseException as exc:
        record['status']='failed';record['failures'].append(dict(error=repr(exc),traceback=traceback.format_exc()));save();raise
    print(json.dumps(dict(status=record['status'],output=str(output),layers=len(record['layers']))),flush=True)

if __name__=='__main__':main()
