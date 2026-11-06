"""Actual eight-GPU native API benchmark for a capped sampled planner.

Only live source-local routes and histograms enter planning. Optional CPU array
hashes are independent correctness oracles, used outside all timings. No expert
weights, token payload, gradients, GEMM or optimizer are timed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import traceback

import numpy as np
import torch
import torch.distributed as dist

from moonep import Buffer, SampledPlannerConfig
from moonep.planning import allocate_planning_outputs, launch_moon_planning, launch_planning
from moonep.inter_rank_sync import launch_inter_rank_sync
from tests.benchmark_sampled_group import (R,S,E,K,FIELDS,read_local_routes,
    precompile_planner,measure,gather_root,plan_snapshot,audit_plans,sha)

ROOT=Path(__file__).resolve().parents[1]


def host(value):
    return value.detach().cpu().numpy().copy()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--candidate-limit',type=int,default=128)
    ap.add_argument('--rounds',type=int,choices=[2,4],default=4)
    ap.add_argument('--expected-hashes',type=Path)
    ap.add_argument('--layers',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--samples',type=int,default=10)
    ap.add_argument('--replays',type=int,default=5)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    expected_hashes=json.loads(args.expected_hashes.read_text()) if args.expected_hashes else None
    cfg=SampledPlannerConfig(candidate_limit=args.candidate_limit,rounds=args.rounds)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    buffer=None;record=None
    def save():
        if rank==0 and record is not None:
            args.output.write_text(json.dumps(record,indent=2)+'\n')
    def verdict(fn):
        result=[None]
        if rank==0:
            try:result[0]={'passed':True,'value':fn()}
            except Exception:result[0]={'passed':False,'traceback':traceback.format_exc()}
        dist.broadcast_object_list(result,src=0)
        if not result[0]['passed']:raise AssertionError(result[0]['traceback'])
        return result[0]['value']
    try:
        torch.cuda.set_device(local);torch.set_num_threads(1)
        dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
        assert dist.get_world_size()==R and int(os.environ['LOCAL_WORLD_SIZE'])==R
        identities=gather_root(dict(rank=rank,host=socket.gethostname(),local_device=local,
            uuid=str(torch.cuda.get_device_properties(local).uuid)))
        def verify_identities():
            assert len({x['uuid'] for x in identities})==R
            return True
        verdict(verify_identities)
        paths=[p for p in (ROOT/'moonep').rglob('*') if p.is_file() and
               p.suffix in ('.py','.cu','.cuh') and not p.name.startswith('._')]
        paths.extend((Path(__file__),Path(__file__).with_name('benchmark_sampled_group.py')))
        record=dict(status='running',metadata=dict(R=R,N=R*S,S=S,E=E,K=K,B=8,
            planner_config=cfg.to_dict(),identities=identities,gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
            samples=args.samples,replays=args.replays,
            oracle_sha256=sha(args.expected_hashes) if args.expected_hashes else None,
            source_files={str(p.relative_to(ROOT)):sha(p) for p in paths},
            timing='CUDA Graph; maximum of rank sample medians; average the same eight layers',
            included=['native seed','live sample collective','candidate scoring/sort/cache pruning',
                'expert selection','full-token exchange and count collectives','physical layout',
                'collective validation and native fallback selection','source metadata publication'],
            excluded=['histogram construction','allocation','JIT','audits','expert/token/gradient payload','GEMM','optimizer']),
            layers=[],failures=[])
        if rank==0:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.touch(exist_ok=False)
        save()
        buffer=Buffer(S,256,K,E,R,B=8,enable_pdl=False,planning_backend='sampled_group',
            planner_config=cfg,explicitly_destroy=True)
        ctx=buffer._require_ctx();precompile_planner(ctx)
        adapter=ctx['sampled_group_planner'];res=adapter.residency
        assert buffer.planner_config==cfg and adapter.config==cfg
        assert res.candidate_limit==args.candidate_limit and adapter.rounds==args.rounds
        native_plan,native_cu=allocate_planning_outputs(ctx)
        plan,cu=allocate_planning_outputs(ctx)
        begin=rank*int(ctx['meta_chunk_padded'])+int(ctx['SRC_INFO_OFF'])
        source_info=ctx['meta_buf'][begin:begin+int(ctx['NvS'])]
        def snapshot():
            item=plan_snapshot(plan,cu,ctx,adapter=adapter)
            item['source_info']=host(source_info)
            item['history']=host(res.history[:int(res.history_count.item())])
            return item
        for layer in map(int,args.layers.split(',')):
            path=args.fixture_root/f'layer_{layer:02d}.npz'
            local_routes=read_local_routes(path,rank)
            routes=torch.from_numpy(local_routes).to('cuda')
            tpe=torch.bincount(routes.flatten().long(),minlength=E).int()
            all_routes=gather_root(local_routes)
            def native():
                launch_inter_rank_sync(ctx)
                launch_moon_planning(ctx,routes.flatten(),tpe,native_cu,native_plan)
            def sampled():
                launch_inter_rank_sync(ctx)
                launch_planning(ctx,routes.flatten(),tpe,cu,plan)
            row=dict(layer=layer,fixture_sha256=sha(path),variants={})
            record['layers'].append(row);save()
            native();torch.cuda.synchronize()
            native_items=gather_root(plan_snapshot(native_plan,native_cu,ctx))
            row['native_audit']=verdict(lambda:audit_plans(native_items,np.stack(all_routes),8))
            sampled();torch.cuda.synchronize();before=snapshot()
            row['variants']['native']=measure(native,args.samples,args.replays)
            row['variants']['sampled']=measure(sampled,args.samples,args.replays)
            # Inspect the final Graph output before any eager execution.
            after=snapshot()
            graph_exact=all(np.array_equal(before[key],after[key]) for key in
                (*FIELDS,'cu','owner','support','source_info','history'))
            graph_checks=gather_root(graph_exact)
            items=gather_root(after);diagnostics=gather_root(res.candidate_diagnostics())
            def check():
                assert all(graph_checks),'Graph output differs from eager'
                assert all(x['status']==0 for x in items),'unexpected native fallback'
                assert all(d==diagnostics[0] for d in diagnostics),'per-rank candidate diagnostics differ'
                assert all(np.array_equal(x['history'],items[0]['history']) for x in items),'per-rank histories differ'
                for original,new in zip(native_items,items):
                    for key in FIELDS:
                        assert np.array_equal(original[key],new['native_seed'][key]),f'native seed {key}'
                    assert np.array_equal(original['cu'],new['native_seed_cu'])
                audit=audit_plans(items,np.stack(all_routes),8,optimized=True)
                assert audit['remote_token_destinations']<=row['native_audit']['remote_token_destinations']
                if expected_hashes is not None:
                    expected=expected_hashes[f'layer_{layer:02d}_cap{args.candidate_limit}']
                    arrays={'support':np.asarray(items[0]['support'],dtype=np.bool_),
                        f'owner_rounds{args.rounds}':np.stack([x['owner'] for x in items]).astype(np.int32),
                        'history':np.asarray(items[0]['history'],dtype=np.int64),
                        'retained_ids':np.sort(np.asarray(diagnostics[0]['selected_ids'],dtype=np.int32))}
                    for key,array in arrays.items():
                        assert list(array.shape)==expected[key]['shape'],f'oracle shape {key}'
                        assert hashlib.sha256(array.tobytes()).hexdigest()==expected[key]['sha256'],f'oracle hash {key}'
                return audit
            row.update(audit=verdict(check),candidate_diagnostics=diagnostics[0] if rank==0 else None,
                graph_outputs_exact=True,all_rank_history_exact=True,cpu_oracle_checked=expected_hashes is not None)
            save()
            if rank==0:print(json.dumps(dict(layer=layer,candidate_limit=args.candidate_limit,passed=True,
                planner_us=row['variants']['sampled']['max_rank_median_cuda_ms']*1000)),flush=True)
        record['status']='complete';save()
    except Exception:
        failure=dict(rank=rank,traceback=traceback.format_exc())
        args.output.with_name(args.output.stem+f'_rank{rank}_failure.json').write_text(json.dumps(failure,indent=2))
        if rank==0 and record is not None:record['status']='failed';record['failures'].append(failure);save()
        raise
    finally:
        if buffer is not None:torch.cuda.synchronize();buffer.destroy()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
