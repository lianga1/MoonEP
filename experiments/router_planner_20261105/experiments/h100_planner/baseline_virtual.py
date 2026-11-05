"""Semantic rt0 logical-8-rank CUDA planner PORT, not native MoonEP kernel.
Output routing/physical/copy/alloc semantics match CPU reference. Single GPU,
no NVLink gather, multicast, remote source writes, cross-rank synchronization.
Counts are precomputed outside timed run like original native benchmark.
"""
from pathlib import Path
import hashlib,os
import torch
from torch.utils.cpp_extension import load
_EXT=None

def extension():
 global _EXT
 if _EXT is None:
  p=Path(__file__).with_suffix('.cu');digest=hashlib.sha256(p.read_bytes()).hexdigest()[:12]
  _EXT=load(name='epmoe_virtual_seed_'+digest,sources=[str(p)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
 return _EXT

class VirtualMoonEPPlanner:
 def __init__(self,routes,B=8,padding=128):
  assert routes.is_cuda and routes.dtype==torch.int32 and routes.is_contiguous() and routes.shape[-1]==8
  self.routes=routes.flatten();self.B=B;self.padding=padding;N=routes.numel();assert N%64==0
  self.NvS=N//8+(padding-1)*88;self.counts=torch.bincount(self.routes.long(),minlength=352).int();device=routes.device
  def arr(*s):return torch.empty(s,dtype=torch.int32,device=device)
  self.ids=torch.arange(N,dtype=torch.int32,device=device);self.sorted_experts=arr(N);self.sorted_ids=arr(N)
  self.alloc=arr(352,8);self.z=arr(8,8);self.starts=arr(352);self.experts_to_copy=arr(8,B);self.offsets=arr(8,352);self.lut=arr(8,352)
  self.cu_seqlens=arr(8,352+B);self.zero_fill=arr(8,352+B,2);self.stats=arr(8,2);self.owner=arr(N//8,8);self.physical_topk=arr(N//8,8);self.dst=arr(N//8,8);self.src_info=arr(8,self.NvS)
  self.ext=extension();nb=self.ext.workspace(self.routes,self.sorted_experts,self.ids,self.sorted_ids);self.tmp=torch.empty(nb,dtype=torch.uint8,device=device)
 def run(self):
  self.ext.run(self.routes,self.counts,self.ids,self.sorted_experts,self.sorted_ids,self.tmp,self.alloc,self.z,self.starts,self.experts_to_copy,self.offsets,self.lut,self.cu_seqlens,self.zero_fill,self.stats,self.owner,self.physical_topk,self.dst,self.src_info,self.B,self.padding)
  return self.owner
 def run_seed(self):
  """Counts-to-owner component only; excludes physical layout/output mapping."""
  self.ext.run_seed(self.routes,self.counts,self.ids,self.sorted_experts,self.sorted_ids,self.tmp,self.alloc,self.z,self.starts,self.owner)
  return self.owner
