"""Correctness and Graph smoke for logical-R8 seed PORT and existing bundle."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from baseline_virtual import VirtualMoonEPPlanner
from bundle_wrapper import BundlePlanner
from bundle_lane8 import Bundle8LanePlanner

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--layers',default='0');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.root/'analysis'));from moonep_cpu import plan,communication
 checks=[]
 for layer in map(int,a.layers.split(',')):
  x=np.load(a.root/f'outputs/trace_audit/layer_{layer:02d}.npz')['topk'].astype(np.int32)
  cpu=plan(x,E=352,B=8,replica_target=0);routes=torch.tensor(x.reshape(-1,8),device='cuda',dtype=torch.int32)
  base=VirtualMoonEPPlanner(routes);base.run();torch.cuda.synchronize()
  assert np.array_equal(base.owner.cpu().numpy().reshape(x.shape),cpu.owner)
  assert np.array_equal(base.alloc.cpu().numpy(),cpu.alloc)
  assert np.array_equal(base.experts_to_copy.cpu().numpy(),cpu.experts_to_copy)
  assert np.array_equal(base.physical_topk.cpu().numpy().reshape(x.shape),cpu.physical_topk)
  raw=base.dst.cpu().numpy();raw=np.where(raw<0,-raw-1,raw)
  assert len(np.unique(raw))==x.size and (raw//base.NvS==cpu.owner.reshape(-1,8)).all()
  src=base.src_info.cpu().numpy().ravel();expected=np.arange(x.size,dtype=np.int64);expected=(expected//(x.shape[1]*8))*base.NvS+expected%(x.shape[1]*8)
  assert np.array_equal(src[raw.ravel()],expected)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):base.run()
  g.replay();torch.cuda.synchronize();assert np.array_equal(base.owner.cpu().numpy().reshape(x.shape),cpu.owner)
  base.run_seed();torch.cuda.synchronize();assert np.array_equal(base.owner.cpu().numpy().reshape(x.shape),cpu.owner)
  sg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(sg):base.run_seed()
  sg.replay();torch.cuda.synchronize();assert np.array_equal(base.owner.cpu().numpy().reshape(x.shape),cpu.owner)
  saved=np.load(a.root/f'outputs/replay/layer{layer:02d}_b8_small256_global_4.npz');support=torch.tensor(saved['support'],device='cuda',dtype=torch.bool);seed=torch.tensor(cpu.owner.reshape(-1,8),device='cuda',dtype=torch.int32)
  bundle=BundlePlanner(routes,support,per_source=x.shape[1]);bundle.run(seed);torch.cuda.synchronize()
  assert np.array_equal(bundle.owner.cpu().numpy().reshape(x.shape),saved['owner'])
  bg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(bg):bundle.run(seed)
  bg.replay();torch.cuda.synchronize();assert np.array_equal(bundle.owner.cpu().numpy().reshape(x.shape),saved['owner'])
  lane8=Bundle8LanePlanner(routes,support,per_source=x.shape[1]);lane8.run(seed);torch.cuda.synchronize()
  assert np.array_equal(lane8.owner.cpu().numpy().reshape(x.shape),saved['owner'])
  lg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(lg):lane8.run(seed)
  lg.replay();torch.cuda.synchronize();assert np.array_equal(lane8.owner.cpu().numpy().reshape(x.shape),saved['owner'])
  checks.append(dict(layer=layer,seed_port_owner_physical_alloc_copies_exact=True,seed_only_owner_exact=True,seed_port_dst_unique=True,seed_port_provenance_exact=True,bundle_cpu_owner_exact=True,bundle_lane8_cpu_owner_exact=True,cuda_graph=True))
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(checks=checks),indent=2));print(checks[-1],flush=True)
if __name__=='__main__':main()
