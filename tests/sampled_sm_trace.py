"""Untimed launch-budget audit; this is not an achieved-occupancy measurement."""
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist


def audit_trace(data,limit):
    kernels=[x for x in data['traceEvents'] if x.get('cat')=='kernel' and x.get('ph')=='X']
    assert kernels,'profiler did not capture CUDA kernels'
    launches=[];events=[]
    for k in kernels:
        args=k.get('args',{});grid=args.get('grid')
        assert isinstance(grid,list) and len(grid)==3,('missing launch dimensions',k.get('name'),args)
        blocks=math.prod(grid)
        launches.append(dict(name=k['name'],grid=grid,blocks=blocks,block=args.get('block'),
            stream=args.get('stream'),duration_us=k['dur']))
        events.extend(((k['ts'],1,blocks),(k['ts']+k['dur'],0,-blocks)))
    live=peak=0
    for _,_,delta in sorted(events):live+=delta;peak=max(peak,live)
    streams=sorted({x['stream'] for x in launches})
    nccl=[x['name'] for x in launches if 'nccl' in x['name'].lower()]
    names=[x['name'] for x in launches]
    anchors={
        'sample_gather':sum('gather_local(' in n for n in names),
        'expert_greedy':sum('global_greedy_table_active<' in n or 'global_greedy_compact_active<' in n for n in names),
        'exchange_proposal':sum('propose_kernel(' in n or 'propose_ordinals_fused(' in n or 'propose_ordinals_fused<' in n for n in names),
        'exchange_prefix':sum('scan_blocks(' in n or 'scan_blocks_warp(' in n for n in names),
        'layout_encode':sum('encode_kernel(' in n or 'bounded_materialize::encode(' in n for n in names),
        'publication':sum('publish_kernel(' in n or 'bounded_materialize::publish(' in n for n in names),
    }
    expected=dict(sample_gather=3,expert_greedy=3,exchange_proposal=12,
                  exchange_prefix=12,layout_encode=3,publication=3)
    assert anchors==expected,('incomplete three-replay capture',anchors,expected)
    passed=(not limit or (max(x['blocks'] for x in launches)<=limit and peak<=limit and not nccl))
    return dict(passed=passed,sm_limit=limit,kernel_count=len(launches),
        max_launch_blocks=max(x['blocks'] for x in launches),max_simultaneous_launched_blocks=peak,
        streams=streams,nccl_kernel_count=len(nccl),anchors=anchors,launches=launches,
        interpretation='Conservative concurrent-CTA upper bound on planner SM use; no SM-ID pinning or achieved-occupancy claim.')


def profile_budget(fn,directory,limit,rank):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'rank{rank}_graph.json'
    packed=path.with_suffix('.json.gz')
    if path.exists() or packed.exists():raise FileExistsError(path)
    for _ in range(3):fn()
    torch.cuda.synchronize();dist.barrier()
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):fn()
        graph.replay();torch.cuda.synchronize();dist.barrier()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            for _ in range(3):graph.replay()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        raw=path.read_bytes();result=audit_trace(json.loads(raw),limit)
        compressed=gzip.compress(raw,compresslevel=6,mtime=0)
        packed.write_bytes(compressed);path.unlink()
        result.update(rank=rank,path=str(packed),sha256=hashlib.sha256(compressed).hexdigest(),
            raw_sha256=hashlib.sha256(raw).hexdigest(),replays=3)
    finally:
        torch.cuda.synchronize();graph.reset()
    dist.barrier()
    return result
