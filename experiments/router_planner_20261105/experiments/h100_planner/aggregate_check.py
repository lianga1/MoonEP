"""Exactness + paired CUDA-event timing for aggregate CUDA port.
Run on selected GPU only, e.g. CUDA_VISIBLE_DEVICES=0 python aggregate_check.py.
Default all eight real layers; all grouping/DP costs included in run().
"""
import argparse,json,time,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from aggregate import AggregateRunner

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--layers',type=int,nargs='+',default=list(range(8)));ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--output',type=Path,default=Path('aggregate_results.json'));ap.add_argument('--tiny',action='store_true');ap.add_argument('--correctness-only',action='store_true');args=ap.parse_args()
    root=args.root;sys.path.insert(0,str(root/'analysis'))
    rows=[]
    if args.tiny:
        tiny=np.load(Path(__file__).with_name('aggregate_tiny.npz'));x=tiny['routes'];owner=tiny['owner'];su=tiny['support'];expected=tiny['expected']
        for mode in (False,True):
            runner=AggregateRunner(torch.tensor(x,device='cuda'),torch.tensor(su,device='cuda'),coalesced=mode)
            result=runner.run(torch.tensor(owner,device='cuda')).cpu().numpy()
            assert np.array_equal(result,expected), {'coalesced':mode,'mismatches':int((result!=expected).sum())}
        print('tiny exact, both layouts');return
    from moonep_cpu import communication
    for layer in args.layers:
        x=np.load(root/f'outputs/trace_audit/layer_{layer:02d}.npz')['topk'].astype(np.int32)
        owner=np.load(root/f'outputs/rt2_b8_comparison/layer{layer:02d}_native_rt0.npz')['owner'].astype(np.int32)
        su=np.load(root/f'outputs/replay/layer{layer:02d}_b8_small256_global_4.npz')['support']
        expected=np.load(root/f'outputs/aggregate_exchange_multipair/layer{layer:02d}_mixed_sizes.npz')['owner']
        xt=torch.tensor(x,device='cuda');ot=torch.tensor(owner,device='cuda');st=torch.tensor(su,device='cuda')
        runners={mode:AggregateRunner(xt,st,coalesced=mode) for mode in (False,True)}
        for mode,runner in runners.items():
            result=runner.run(ot).cpu().numpy()
            if not np.array_equal(result,expected):
                np.savez_compressed(args.output.parent/f'aggregate_failure_layer{layer:02d}_soa{int(mode)}.npz',owner=result,expected=expected)
                raise AssertionError(f'layer {layer} coalesced {mode}: {(result!=expected).sum()} mismatched pairs')
            assert np.all(su[result,x])
            assert np.array_equal(np.bincount(result.ravel(),minlength=8),np.bincount(owner.ravel(),minlength=8))
        times={False:[],True:[]};walls={False:[],True:[]}
        for rep in range(0 if args.correctness_only else args.repeats):
            for mode in ((False,True) if rep%2==0 else (True,False)):
                torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record();r=runners[mode].run(ot);end.record();end.synchronize();walls[mode].append(1e3*(time.perf_counter()-wall));times[mode].append(start.elapsed_time(end))
                assert np.array_equal(r.cpu().numpy(),expected)
        rows.append(dict(layer=layer,exact=True,communication=communication(x,expected,352),event_ms={str(k):v for k,v in times.items()},host_wall_ms={str(k):v for k,v in walls.items()}))
        args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(dict(scope='eager CUDA extension; all four round GPU grouping, prefix, mixed DP, expansion included; fixed resident input and seed excluded; Thrust sorting may synchronize host; no CPU grouping/tensor readback in timed run',gpu=torch.cuda.get_device_name(),torch=torch.__version__,rows=rows),indent=2))
        print(layer,{str(k):(float(np.median(v)) if v else None) for k,v in times.items()},flush=True)

if __name__=='__main__':main()
