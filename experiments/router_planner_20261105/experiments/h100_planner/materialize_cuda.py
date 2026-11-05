"""Fixed-residency GPU output builder; stable rows and compact physical-slot layout.
Slots fixed outside timing; prepare/sort/boundaries/padding/mapping/dedup inside.
All buffers preallocated. cu_seqlens and zero_fill use compact [R,44+B] slots.
"""
from pathlib import Path
import hashlib,torch
from torch.utils.cpp_extension import load
_EXT=None
class CudaMaterializer:
 def __init__(self,routes,support,B=8,padding=128):
  global _EXT
  self.routes=routes.reshape(-1).contiguous();dev=routes.device;self.M=routes.numel();self.Q=44+B;self.cap=self.M//8;self.NvS=self.cap+127*88;self.padding=padding
  assert routes.dtype==torch.int32 and self.M%64==0
  e=torch.arange(352,device=dev);r=torch.arange(8,device=dev)[:,None];extra=support&(e[None,:]//44!=r)
  assert int(extra.sum(1).max())<=B
  self.lut=torch.where(e[None,:]//44==r,e[None,:]%44,torch.where(extra,44+extra.int().cumsum(1)-1,-1)).int()
  if _EXT is None:
   p=Path(__file__).with_suffix('.cu');digest=hashlib.sha256(p.read_bytes()).hexdigest()[:12]
   _EXT=load(name='epmoe_materialize_'+digest,sources=[str(p)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
  self.ext=_EXT
  def a(*s):return torch.empty(s,dtype=torch.int32,device=dev)
  self.ids=torch.arange(self.M,device=dev,dtype=torch.int32);self.physical=a(self.M);self.sorted_keys=a(self.M);self.sorted_ids=a(self.M)
  self.starts=a(8*self.Q);self.counts=a(8,self.Q);self.offsets=a(8*self.Q);self.cu=a(8,self.Q);self.zfr=a(8,self.Q,2)
  self.raw=a(self.M);self.dst=a(self.M//8,8);self.provenance=a(8*self.NvS)
  n=self.ext.workspace(self.physical,self.sorted_keys,self.ids,self.sorted_ids)
  self.temp=torch.empty(n,dtype=torch.uint8,device=dev)
  self.result=dict(dst=self.dst,physical_topk=self.physical,counts=self.counts,cu_seqlens=self.cu,zero_fill=self.zfr,provenance=self.provenance,raw_dst=self.raw)
 def run(self,owner):
  self.ext.run(self.routes,owner,self.lut,self.ids,self.physical,self.sorted_keys,self.sorted_ids,self.temp,self.starts,self.counts,self.offsets,self.cu,self.zfr,self.raw,self.dst,self.provenance,self.Q,self.padding)
  return self.result
