"""Run on CUDA host; verifies exact sequential and simultaneous outputs.
Usage: python experiments/h100_planner/subset_smoke.py --root PATH --layers 0
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from subset_cuda import SubsetRunner
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--layers',type=int,nargs='+',default=list(range(8)));ap.add_argument('--repeats',type=int,default=10);ap.add_argument('--correctness-only',action='store_true');ap.add_argument('--variants',nargs='+',default=['sequential','parallel','indexed']);args=ap.parse_args()
root=args.root;rows=[]
for l in args.layers:
 x=np.load(root/f'outputs/trace_audit/layer_{l:02d}.npz')['topk'].astype(np.int32)
 old=np.load(root/f'outputs/rt2_b8_comparison/layer{l:02d}_native_rt0.npz')['owner'].astype(np.int32)
 su=np.load(root/f'outputs/replay/layer{l:02d}_b8_small256_global_4.npz')['support']
 xg=torch.tensor(x,device='cuda');og=torch.tensor(old,device='cuda');sg=torch.tensor(su,device='cuda')
 runner=SubsetRunner(xg,sg,h=2,g=8)
 for variant in args.variants:
  fn={'sequential':runner.full,'parallel':runner.full_parallel,'indexed':runner.full_indexed}[variant]
  target=root/f'outputs/pruned_subset_indexed/layer{l:02d}_h2_g8.npz' if variant=='sequential' else root/f'outputs/h100_subset_reference/layer{l:02d}.npz'
  expected=np.load(target)['owner'];got=fn(og).cpu().numpy()
  if not np.array_equal(got,expected):
   where=np.argwhere(got!=expected);print('MISMATCH',l,variant,len(where),where[:10],flush=True)
   np.savez_compressed(root/f'subset_mismatch_layer{l}_{variant}.npz',got=got,expected=expected)
   raise AssertionError('GPU vs CPU mismatch')
  for _ in range(3):fn(og)
  torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):fn(og)
  times=[]
  graph.replay()
  for _ in range(0 if args.correctness_only else args.repeats):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record();graph.replay();end.record();end.synchronize();times.append(start.elapsed_time(end)*1000)
  assert np.array_equal(runner.output.cpu().numpy(),expected)
  row=dict(layer=l,variant=variant,bitwise=True,graph_bitwise=True,graph_us=times,median_us=float(np.median(times)) if times else None)
  print(json.dumps(row),flush=True);rows.append(row)
(root/'subset_smoke_results.json').write_text(json.dumps({'rows':rows,'scope':'full GPU subset component including shortlist mining, excludes seed/support generation and physical plan materialization'},indent=2))
