"""Exact checks and paired two-GPU Graph timings for residency variants.

Run with torchrun --standalone --nproc_per_node=2. All eight real folded
layers and separately labelled synthetic correctness cases are checked.
Only real layers are timed. Full model, native planner/owner decode, expert
copies and token dispatch/combine are outside these component measurements.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from residency_cuda import TwoRankResidency as Base
from residency_cuda_optimized import TwoRankResidency as Optimized

ROOT=Path(__file__).resolve().parents[2]


def collect(value):
    out=[None]*dist.get_world_size();dist.all_gather_object(out,value);return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,default=ROOT/'outputs/two_gpu_overlap/inputs')
    ap.add_argument('--corner-dir',type=Path,default=ROOT/'outputs/two_gpu_overlap/residency_corners')
    ap.add_argument('--layers',type=int,nargs='+',default=list(range(8)))
    ap.add_argument('--variants',nargs='+',choices=['tops','warp','combined','control','fused','fused_warp'],default=['tops','warp','combined'])
    ap.add_argument('--samples',type=int,default=10)
    ap.add_argument('--repeats',type=int,default=10)
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15));assert dist.get_world_size()==2
    exists=[args.output.exists() if rank==0 else None];dist.broadcast_object_list(exists,src=0)
    if exists[0]:raise FileExistsError('Use a fresh output filename; preserve prior attempts')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',metadata=dict(created_at=datetime.now(timezone.utc).isoformat(),world=2,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        variants=['base',*args.variants],samples=args.samples,repeats=args.repeats,warmup=args.warmup,
        timing='unprofiled paired Graph component cohort; per-round max rank CUDA event time; original full benchmark remains separate',
        scopes=dict(select='GPU selection after sample gathering',gather_select='runtime route+owner sample gathering, collective, GPU selection'),
        excluded='native owner decoding, native planner, physical address building, expert copies, token communication, expert compute',
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('residency_cuda*') if p.suffix in ('.py','.cu')}),
        real_layers=[],synthetic_checks=[],failures=[])
    def save():
        if rank==0:args.output.write_text(json.dumps(record,indent=2)+'\n')
    save();graphs={}
    real=json.loads((args.input_dir/'manifest.json').read_text())
    ref_real={x['layer']:x for x in real['layer_records']}
    corner=json.loads((args.corner_dir/'manifest.json').read_text())
    cases=[]
    for c in corner['cases']:
        cases.append(dict(name=c['name'],path=args.corner_dir/c['file'],synthetic=True,history=c['history'],diagnostics=c['diagnostics']))
    for layer in args.layers:
        d=ref_real[layer]['residency']
        history=[[h['a'],h['b'],len(h['add_a']),len(h['add_b']),h['size'],h['estimated_sample_gain'],len(h['add_a'])+len(h['add_b'])] for h in d['accepted_packages']]
        cases.append(dict(name=f'layer{layer:02d}',layer=layer,path=args.input_dir/f'layer_{layer:02d}.npz',synthetic=False,
                          history=history,diagnostics=[d['candidate_packages'],*d['candidate_extra']]))
    try:
        for case in cases:
            raw=np.load(case['path'],allow_pickle=False);S=raw['topk'].shape[1]
            def gpu(a,dtype=torch.int32):return torch.tensor(a,device='cuda',dtype=dtype)
            local_routes,owner=gpu(raw['topk'][rank]),gpu(raw['owner'][rank]);etc=gpu(raw['native_experts_to_copy'])
            ctor=dict(local_tokens=S,rank=rank,slots=8)
            sample_routes,ids=gpu(raw['sample_routes']),gpu(raw['sample_ids'],torch.int64)
            runners={'base':Base(sample_routes,ids,**ctor)}
            for name in args.variants:
                if name.startswith('fused'):
                    from residency_cuda_fused import TwoRankResidency as Fused
                    runners[name]=Fused(sample_routes,ids,variant=name,**ctor)
                else:
                    runners[name]=Optimized(sample_routes,ids,variant=name,**ctor)
            for runner in runners.values():runner.run(etc,owner,local_routes=local_routes)
            # Repeat before rank-local assertions so failures cannot strand the
            # other rank inside a subsequent collective in the assertion block.
            for runner in runners.values():runner.run(etc,owner,local_routes=local_routes)
            torch.cuda.synchronize()
            error=None;local_audits={}
            try:
                base=runners['base'];valid=base.classes>=0
                for name,runner in runners.items():
                    count=int(runner.history_count.item())
                    assert np.array_equal(runner.support.cpu().numpy(),raw['support']),(name,'CPU support')
                    assert runner.history[:count].cpu().tolist()==case['history'],(name,'CPU history')
                    assert runner.diagnostics.cpu().tolist()==case['diagnostics'],(name,'CPU diagnostics')
                    for field in ('support','classes','sig0','sig1','frequencies','tops','first','experts_to_copy','extra'):
                        assert torch.equal(getattr(runner,field),getattr(base,field)),(name,field)
                    assert torch.equal(runner.groups[valid],base.groups[valid]),(name,'valid groups')
                    assert np.array_equal(runner.routes.cpu().numpy(),raw['sample_routes']),(name,'joint routes')
                    assert np.array_equal(runner.sample_owner.cpu().numpy(),raw['owner'].reshape(-1,8)[raw['sample_ids']]),(name,'joint owner')
                    local_audits[name]=dict(exact_cpu_support_history_diagnostics=True,exact_all_intermediates=True,exact_joint_samples=True,repeat=True)
            except BaseException as exc:
                error=dict(case=case['name'],rank=rank,error=repr(exc),traceback=traceback.format_exc())
            errors=[x for x in collect(error) if x is not None]
            if errors:record['failures'].extend(errors);raise RuntimeError('Residency candidate validation failed')
            row=dict(name=case['name'],synthetic=case['synthetic'],shape=list(raw['topk'].shape),samples=len(raw['sample_ids']),
                input_sha256=hashlib.sha256(case['path'].read_bytes()).hexdigest(),diagnostics=case['diagnostics'],
                per_rank_audits=collect(local_audits))
            record['synthetic_checks' if case['synthetic'] else 'real_layers'].append(row);save()
            if case['synthetic'] or args.validate_only:
                if rank==0:print(case['name']+': exact validation passed',flush=True)
                continue
            functions={}
            for name,runner in runners.items():
                functions[name+'/select']=lambda runner=runner:runner.select(etc)
                functions[name+'/gather_select']=lambda runner=runner:runner.run(etc,owner,local_routes=local_routes)
            row['timings']={key:[] for key in functions}
            for name,fn in functions.items():
                for _ in range(args.warmup):fn()
                torch.cuda.synchronize();dist.barrier()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):fn()
                graphs[name]=graph;graph.replay();torch.cuda.synchronize();dist.barrier()
            names=list(functions)
            for sample in range(args.samples):
                order=names[sample%len(names):]+names[:sample%len(names)]
                for name in order:
                    dist.barrier();torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();tick=time.perf_counter()
                    for _ in range(args.repeats):graphs[name].replay()
                    end.record();end.synchronize()
                    measured=collect(dict(cuda_ms=start.elapsed_time(end)/args.repeats,
                        host_ms=(time.perf_counter()-tick)*1000/args.repeats))
                    row['timings'][name].append(dict(sample=sample,per_rank=measured,
                        max_cuda_ms=max(r['cuda_ms'] for r in measured),max_host_ms=max(r['host_ms'] for r in measured)))
                save()
            row['median_cuda_ms']={name:statistics.median(x['max_cuda_ms'] for x in rounds) for name,rounds in row['timings'].items()}
            for name,runner in runners.items():
                assert np.array_equal(runner.support.cpu().numpy(),raw['support']),(name,'Graph support')
                assert runner.history[:int(runner.history_count.item())].cpu().tolist()==case['history'],(name,'Graph history')
            row['post_graph_exact']=True
            for graph in graphs.values():graph.reset()
            graphs.clear();graph=None;fn=None
            torch.cuda.synchronize();dist.barrier();save()
            if rank==0:print(json.dumps({'case':case['name'],'median_cuda_ms':row['median_cuda_ms']}),flush=True)
        record['status']='complete';save()
    except BaseException as exc:
        record['status']='failed';record['failures'].append(dict(rank=rank,error=repr(exc),traceback=traceback.format_exc()));save();raise
    finally:
        for graph in graphs.values():graph.reset()
        graphs.clear();graph=None
        torch.cuda.synchronize()
        dist.destroy_process_group()


if __name__=='__main__':main()
