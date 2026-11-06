"""Private frequency-sharding experiment; production select remains unchanged.

CPU index/hash-collision checks:
  python -m tests.benchmark_frequency_sharding --cpu-self-test --output result.json
Eight actual GPUs, live source-local routes and native seed:
  torchrun --standalone --nproc_per_node=8 -m tests.benchmark_frequency_sharding \
    --fixture-root /path/to/kernel_inputs --output /path/to/fresh_result.json

The useful comparison is local_old versus shard_complete, which includes the
new peer all-gather and unpack. shard_compute processes only one eighth of the
records and must never be reported as a complete-stage speedup. Explicit GPU
pre-sync precedes every event start and is excluded from each interval. Two
whole Graph warmups precede samples. There is no CPU barrier before samples.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_self_test():
    """Independent tuple oracle versus lane partition/reduction and unpack."""
    rows = []
    for d, k in [(1, 1), (7, 3), (37, 3), (256, 8), (257, 8), (512, 8)]:
        for collision in (False, True):
            rng = np.random.default_rng(1031 + d + collision)
            records, expected = [], []
            for pair in range(28):
                prototypes = []
                for p in range(12):
                    size = 1 + p % k
                    group = tuple(sorted(rng.choice(64, size, replace=False).tolist()))
                    cls = ((p % 2) * (k + 1) + size) * 2 + (p // 2) % 2
                    prototypes.append((cls, group, 0 if collision else p,
                                       0 if collision else p * 17))
                choices = rng.integers(0, len(prototypes), size=d)
                own = [prototypes[int(v)] for v in choices]
                # Exercise the first leader, later duplicates, and invalid records.
                own[0] = prototypes[0]
                if d > 1:
                    own[-1] = own[0]
                if d > 2:
                    own[1] = (-1, (), 0, 0)
                seen, truth = {}, [0] * d
                for i, (cls, group, _, _) in enumerate(own):
                    if cls >= 0:
                        seen.setdefault((cls, group), []).append(i)
                for ids in seen.values():
                    truth[ids[0]] = len(ids)
                records.extend(own)
                expected.extend(truth)
            words = (len(records) + 7) // 8
            packed = [[-99] * words for _ in range(8)]
            for rank in range(8):
                for out in range(words):
                    idx = out * 8 + rank
                    if idx >= len(records):
                        packed[rank][out] = 0
                        continue
                    cls, group, h0, h1 = records[idx]
                    base, own = idx // d * d, idx % d
                    counts, first = [0] * 32, [d] * 32
                    if cls >= 0:
                        for lane in range(32):
                            for j in range(lane, d, 32):
                                other = records[base + j]
                                if other[0] == cls and other[2:] == (h0, h1) and other[1] == group:
                                    counts[lane] += 1
                                    first[lane] = min(first[lane], j)
                    for delta in (16, 8, 4, 2, 1):
                        old_count, old_first = counts[:], first[:]
                        for lane in range(32 - delta):
                            counts[lane] += old_count[lane + delta]
                            first[lane] = min(old_first[lane], old_first[lane + delta])
                    packed[rank][out] = counts[0] if first[0] == own else 0
            actual = [packed[i % 8][i // 8] for i in range(len(records))]
            assert actual == expected, (d, k, collision)
            assert all(packed[r][i] == 0 for r in range(8) for i in range(words)
                       if i * 8 + r >= len(records))
            rows.append(dict(D=d, K=k, forced_hash_collision=collision,
                             flat_records=len(records), shard_words=words,
                             first_occurrence_exact=True, padded_tail_zero=True,
                             expected_sha256=hashlib.sha256(np.asarray(expected, np.int32).tobytes()).hexdigest()))
    return dict(status='complete',scope='CPU arithmetic/index oracle; not CUDA execution',cases=rows)


def gpu_main(args):
    import torch
    import torch.distributed as dist
    from moonep import Buffer, SampledPlannerConfig
    from moonep.inter_rank_sync import launch_inter_rank_sync
    from moonep.planning import launch_moon_planning
    from tests.benchmark_sampled_group import R, S, E, K, read_local_routes, precompile_planner, gather_root

    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    buffer, record = None, None

    def save():
        if rank == 0 and record is not None:
            args.output.write_text(json.dumps(record, indent=2) + '\n')

    def all_check(local_value, label):
        values = gather_root(bool(local_value))
        verdict = [all(values) if rank == 0 else None]
        dist.broadcast_object_list(verdict, src=0)
        if not verdict[0]:
            raise AssertionError(label + ': at least one rank failed')
        return values

    def tensor_sha(tensor):
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

    def measure(fn, ctx):
        # Event nodes, not host launch time, delimit each invocation. Pre-sync
        # stays before BEGIN on the same stream, including during Graph replay.
        starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(args.replays)]
        ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(args.replays)]
        for a, b in zip(starts, ends):
            a.record(); b.record()
        ends[-1].synchronize()
        # The single setup barrier is outside capture, warmup, and all samples.
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        raw = []
        try:
            with torch.cuda.graph(graph):
                for a, b in zip(starts, ends):
                    launch_inter_rank_sync(ctx)
                    a.record()
                    fn()
                    b.record()
            for _ in range(2):
                graph.replay()
                ends[-1].synchronize()
            for sample in range(args.samples):
                tick = time.perf_counter()
                graph.replay()
                ends[-1].synchronize()
                host_ms = (time.perf_counter() - tick) * 1000 / args.replays
                values = [a.elapsed_time(b) for a, b in zip(starts, ends)]
                item = dict(rank=rank, replay_cuda_ms=values,
                            cuda_ms=statistics.mean(values), host_ms_including_presync=host_ms)
                ranks = gather_root(item)
                if rank == 0:
                    raw.append(dict(sample=sample, replays=args.replays, per_rank=ranks,
                                    max_rank_cuda_ms=max(x['cuda_ms'] for x in ranks)))
            if rank != 0:
                return None
            medians = [dict(rank=r, cuda_ms=statistics.median(
                row['per_rank'][r]['cuda_ms'] for row in raw)) for r in range(R)]
            return dict(raw=raw, per_rank_medians=medians,
                        max_rank_median_cuda_ms=max(x['cuda_ms'] for x in medians),
                        graph_warmups=2, presync='GPU barrier before every start event; excluded',
                        cpu_barrier_before_samples=False)
        finally:
            graph.reset()

    try:
        torch.cuda.set_device(local); torch.set_num_threads(1)
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
        assert dist.get_world_size() == R and int(os.environ['LOCAL_WORLD_SIZE']) == R
        identities = gather_root(dict(rank=rank, host=socket.gethostname(), local_device=local,
            uuid=str(torch.cuda.get_device_properties(local).uuid)))
        topology = [None]
        if rank == 0:
            topology[0] = len({x['uuid'] for x in identities}) == 8 and len({x['host'] for x in identities}) == 1
        dist.broadcast_object_list(topology, src=0)
        assert topology[0], 'eight distinct GPUs on one host required'
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.touch(exist_ok=False)
        cfg = SampledPlannerConfig(candidate_limit=128, rounds=4,
                                   sm_limit=args.sm_limit, fuse=args.fuse)
        paths = sorted(p for p in (ROOT / 'moonep').rglob('*')
                       if p.is_file() and p.suffix in ('.py', '.cu', '.cuh') and not p.name.startswith('._'))
        paths.extend([Path(__file__), ROOT / 'tests/benchmark_sampled_group.py'])
        record = dict(status='running', metadata=dict(R=R,S=S,N=R*S,E=E,K=K,B=8,
            planner_config=cfg.to_dict(),identities=identities,gpu=torch.cuda.get_device_name(),
            device_sms=torch.cuda.get_device_properties(local).multi_processor_count,
            torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
            samples=args.samples,replays=args.replays,graph_warmups=2,
            source_files={str(p.relative_to(ROOT)):file_sha(p) for p in paths},
            timing='maximum of per-rank medians of per-round mean replay CUDA intervals; pre-sync excluded',
            decisive_comparison='local_old versus shard_complete; the latter includes all_gather and unpack',
            input_boundary='live source-local topk only; native seed and global sample packet built on GPUs',
            excluded=['record construction','top-g','candidate sorting','greedy','token exchange','payload transfers','GEMM','optimizer']),
            layers=[],failures=[])
        save()
        buffer = Buffer(S,256,K,E,R,B=8,enable_pdl=False,planning_backend='sampled_group',
                        planner_config=cfg,explicitly_destroy=True)
        ctx = buffer._require_ctx();adapter = ctx['sampled_group_planner'];res = adapter.residency
        runtime = adapter.runtime
        precompile_planner(adapter.planning_ctx)
        words = (res.frequencies.numel() + 7) // 8
        assert words <= runtime.capacity
        packed = torch.empty(words,device='cuda',dtype=torch.int32)
        gathered = torch.empty((R,words),device='cuda',dtype=torch.int32)
        record['metadata'].update(sample_D=res.sample_size,shard_words=words,
            per_rank_collective_input_bytes=words*4,per_rank_gather_output_bytes=R*words*4,
            records_per_rank_compute=words,full_records=res.frequencies.numel())

        def shard_compute():
            res.diagnostic_frequency_shard(packed)

        def shard_complete():
            shard_compute()
            runtime.all_gather(packed,gathered)
            res.diagnostic_frequency_unshard(gathered)

        variants = [('local_old',lambda:res.diagnostic_frequency_local(warp=False)),
                    ('local_warp',lambda:res.diagnostic_frequency_local(warp=True)),
                    ('shard_compute',shard_compute),('shard_complete',shard_complete)]
        for layer in map(int,args.layers.split(',')):
            path = args.fixture_root / f'layer_{layer:02d}.npz'
            local_routes = read_local_routes(path,rank)
            routes = torch.from_numpy(local_routes).to('cuda')
            tpe = torch.bincount(routes.flatten().long(),minlength=E).int()
            # Only this source's routes are loaded. Fixture seed/support arrays
            # are never consumed, including to prepare the measured candidates.
            runtime.copy(routes,adapter.routes)
            launch_inter_rank_sync(ctx)
            launch_moon_planning(adapter.planning_ctx,routes.flatten(),tpe,adapter.seed_cu,adapter.seed)
            adapter.materializer.decode_owner(adapter.seed.dst,adapter.seed_owner)
            res.run(adapter.routes,adapter.seed_owner,adapter.seed.experts_to_copy)
            torch.cuda.synchronize()
            all_check(int(res.status.item())==0,'native sample/record preparation')
            expected = res.frequencies.clone()
            expected_packed = torch.zeros_like(packed)
            shard = expected.view(-1)[rank::8]
            expected_packed[:shard.numel()].copy_(shard)
            hashes = gather_root(dict(packet=tensor_sha(res.packet),classes=tensor_sha(res.classes),
                                     groups=tensor_sha(res.groups),sig0=tensor_sha(res.sig0),
                                     sig1=tensor_sha(res.sig1),frequencies=tensor_sha(expected)))
            same = [all(x==hashes[0] for x in hashes) if rank==0 else None]
            dist.broadcast_object_list(same,src=0)
            assert same[0], 'all ranks must share the same complete records'
            row = dict(layer=layer,fixture_sha256=file_sha(path),input_hashes_by_rank=hashes,
                       variants={},exact_checks={})
            if rank==0:
                record['layers'].append(row);save()
            for name,fn in variants:
                packed.fill_(-99);gathered.fill_(-99);res.frequencies.fill_(-99)
                launch_inter_rank_sync(ctx);fn();torch.cuda.synchronize()
                ok = (torch.equal(packed,expected_packed) if name=='shard_compute'
                      else torch.equal(res.frequencies,expected))
                all_check(ok,name+' eager frequency')
                result=measure(fn,ctx)
                ok = (torch.equal(packed,expected_packed) if name=='shard_compute'
                      else torch.equal(res.frequencies,expected))
                checks=all_check(ok,name+' Graph frequency')
                if rank==0:
                    row['variants'][name]=result;row['exact_checks'][name]=checks;save()
            if rank==0:
                old=row['variants']['local_old']['max_rank_median_cuda_ms']
                complete=row['variants']['shard_complete']['max_rank_median_cuda_ms']
                row['shard_complete_speedup_vs_local_old']=old/complete
                print(json.dumps(dict(layer=layer,passed=True,local_old_us=old*1000,
                    shard_complete_us=complete*1000,complete_speedup=old/complete)),flush=True)
                save()
        # Direct-record adversarial GPU fixtures supplement the live-route
        # measurements. D=37 gives 1036 records: each rank receives 130 words,
        # including four padded outputs. All hashes collide in the second
        # fixture while distinct exact expert tuples must remain distinct.
        from moonep.sampled_residency import Residency
        synth = Residency(dict(R=8,E=64,K=3,S=64,B=8,rank=rank,device=torch.device('cuda',local)),
            sample_size=37,groups_per_class=2,candidate_limit=128,component_only=True,
            sm_limit=args.sm_limit,fuse=args.fuse)
        torch.cuda.synchronize();dist.barrier()  # JIT completion only, never per timing sample.
        sd,sk=37,3
        sc=np.full((28,sd),-1,np.int32)
        sg=np.full((28,sd,sk),-1,np.int32)
        sh0=np.zeros((28,sd),np.int64);sh1=np.zeros_like(sh0)
        truth=np.zeros((28,sd),np.int32)
        for pair in range(28):
            prototypes=[tuple(sorted(((pair*7+z*3)%64,(pair*7+z*3+1)%64))) for z in range(4)]
            leaders={}
            for token in range(sd):
                if token==1:continue
                choice=0 if token==sd-1 else token%4
                group=prototypes[choice];sc[pair,token]=5  # dir=0,size=2,gain=1
                sg[pair,token,:2]=group;sh0[pair,token]=choice+1;sh1[pair,token]=(choice+1)*17
                leaders.setdefault(group,[]).append(token)
            for positions in leaders.values():truth[pair,positions[0]]=len(positions)
        synth.classes.copy_(torch.from_numpy(sc).to('cuda'))
        synth.groups.copy_(torch.from_numpy(sg).to('cuda'))
        s_expected=torch.from_numpy(truth).to('cuda')
        s_words=(synth.frequencies.numel()+7)//8
        s_packed=torch.empty(s_words,device='cuda',dtype=torch.int32)
        s_gathered=torch.empty((8,s_words),device='cuda',dtype=torch.int32)
        s_expected_packed=torch.zeros_like(s_packed)
        s_view=s_expected.view(-1)[rank::8]
        s_expected_packed[:s_view.numel()].copy_(s_view)
        def synthetic_shard():
            synth.diagnostic_frequency_shard(s_packed)
        def synthetic_complete():
            synthetic_shard();runtime.all_gather(s_packed,s_gathered)
            synth.diagnostic_frequency_unshard(s_gathered)
        synthetic_variants=[('local_old',lambda:synth.diagnostic_frequency_local(warp=False)),
            ('local_warp',lambda:synth.diagnostic_frequency_local(warp=True)),
            ('shard_compute',synthetic_shard),('shard_complete',synthetic_complete)]
        record['synthetic_checks']=[]
        for collision in (False,True):
            synth.sig0.copy_(torch.from_numpy(np.zeros_like(sh0) if collision else sh0).to('cuda'))
            synth.sig1.copy_(torch.from_numpy(np.zeros_like(sh1) if collision else sh1).to('cuda'))
            item=dict(kind='D37_exact_groups',D=sd,K=sk,E=64,forced_hash_collision=collision,
                full_records=28*sd,shard_words=s_words,padded_words_total=8*s_words-28*sd,
                expected_sha256=tensor_sha(s_expected),variants={},exact_checks={})
            if rank==0:record['synthetic_checks'].append(item);save()
            for name,fn in synthetic_variants:
                s_packed.fill_(-77);s_gathered.fill_(-77);synth.frequencies.fill_(-77)
                launch_inter_rank_sync(ctx);fn();torch.cuda.synchronize()
                ok=(torch.equal(s_packed,s_expected_packed) if name=='shard_compute'
                    else torch.equal(synth.frequencies,s_expected))
                all_check(ok,'D37 collision='+str(collision)+' '+name+' eager')
                result=measure(fn,ctx)
                ok=(torch.equal(s_packed,s_expected_packed) if name=='shard_compute'
                    else torch.equal(synth.frequencies,s_expected))
                checks=all_check(ok,'D37 collision='+str(collision)+' '+name+' Graph')
                if rank==0:item['variants'][name]=result;item['exact_checks'][name]=checks;save()
            if rank==0:print(json.dumps(dict(synthetic_D37=True,collision=collision,passed=True)),flush=True)
        if rank==0:
            summary={name:statistics.mean(row['variants'][name]['max_rank_median_cuda_ms']
                     for row in record['layers'])*1000 for name,_ in variants}
            record['summary']=dict(mean_layer_us=summary,
                complete_speedup_vs_local_old=summary['local_old']/summary['shard_complete'],
                all_frequency_entries_exact=True,production_sharding_enabled=False)
            record['status']='complete';save()
    except Exception:
        failure=dict(rank=rank,traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_name(args.output.stem+f'_rank{rank}_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        if rank==0 and record is not None:
            record['status']='failed';record['failures'].append(failure);save()
        raise
    finally:
        if buffer is not None:
            torch.cuda.synchronize();buffer.destroy()
        if dist.is_initialized():dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture-root',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cpu-self-test',action='store_true')
    ap.add_argument('--sm-limit',type=int,default=32)
    ap.add_argument('--fuse',action='store_true')
    ap.add_argument('--layers',default='0,1,2,3,4,5,6,7')
    ap.add_argument('--samples',type=int,default=10)
    ap.add_argument('--replays',type=int,default=5)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.samples<1 or args.replays<1 or args.sm_limit<1:ap.error('positive samples/replays/sm-limit required')
    if args.cpu_self_test:
        result=cpu_self_test();result['test_source_sha256']=file_sha(__file__)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status=result['status'],cases=len(result['cases']))))
    else:
        if args.fixture_root is None:ap.error('--fixture-root is required for GPU testing')
        gpu_main(args)


if __name__=='__main__':main()
