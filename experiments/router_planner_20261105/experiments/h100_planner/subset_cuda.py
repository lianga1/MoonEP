"""GPU subset planner. Entire mining/shortlisting/exchange stays on GPU.
Eight virtual routing ranks on one GPU. CPU loops only enqueue fixed kernels.
"""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
MOD=None

def extension():
 global MOD
 if MOD is None:
  MOD=load(name='epmoe_subset_h100',sources=[str(Path(__file__).with_suffix('.cu'))],extra_cuda_cflags=['-O3','-lineinfo'],extra_cflags=['-O3'],verbose=True)
 return MOD

class SubsetRunner:
 def __init__(self,routes,support,h=2,g=8):
  assert routes.dtype==torch.int32 and routes.is_cuda and routes.is_contiguous()
  assert tuple(support.shape)==(8,352) and support.dtype==torch.bool
  assert int(support.sum(1).max().item())<=52, 'fixed 44 home + 8 extra capacity'
  self.mod=extension();self.routes=routes;self.support=support.contiguous();self.h=h;self.g=g
  self.T=routes.numel()//8;T=self.T;dev=routes.device
  def alloc(shape,dtype=torch.int32):return torch.empty(shape,device=dev,dtype=dtype)
  self.ex=alloc((8,T));self.ow=alloc((8,T));self.res=alloc(352)
  self.lut=alloc((8,352));self.slots=alloc((8,52))
  self.counts=alloc(256*(52+52*52));self.top=alloc(256*g)
  self.patterns=alloc((64,128));self.lengths=alloc(64)
  self.used=alloc(T,torch.uint8);self.keys0=alloc(T,torch.int64);self.keys1=alloc(T,torch.int64)
  self.masks=alloc(T);self.hist=alloc(12);self.accept=alloc(12);self.offsets=alloc(12)
  self.scores=alloc((2*T+256)//256,torch.int64);self.temp=alloc(self.mod.temp_bytes(T),torch.uint8)
  self.flat=alloc(2048);self.nflat=alloc(1)
  self.member=alloc(64*(52+52*52))
  self.phist=alloc(64*12);self.paccept=alloc(64*12);self.poffsets=alloc(64*12)
  self.output=alloc(routes.shape)
 def prepare(self,owner):
  assert owner.dtype==torch.int32 and owner.is_contiguous()
  self.mod.prepare(self.routes,owner,self.support,self.ex,self.ow,self.res,self.lut,self.slots,self.counts,self.top,self.patterns,self.lengths,self.h,self.g)
  self.mod.flatten(self.patterns,self.lengths,self.flat,self.nflat)
 def run(self,owner,rounds=4):
  assert owner.dtype==torch.int32 and owner.is_contiguous()
  self.mod.run(owner,self.ex,self.ow,self.patterns,self.lengths,self.used,self.keys0,self.keys1,self.masks,self.hist,self.accept,self.offsets,self.scores,self.temp,self.output,rounds)
  return self.output
 def full(self,owner,rounds=4):
  self.prepare(owner);return self.run(owner,rounds)

 def run_parallel(self,owner,rounds=4,policy=1):
  assert owner.dtype==torch.int32 and owner.is_contiguous()
  self.mod.run_parallel(owner,self.ex,self.ow,self.flat,self.nflat,self.keys0,self.keys1,self.masks,self.phist,self.paccept,self.poffsets,self.temp,self.output,rounds,policy)
  return self.output
 def full_parallel(self,owner,rounds=4,policy=1):
  self.prepare(owner);return self.run_parallel(owner,rounds,policy)
 def prepare_indexed(self,owner):
  self.prepare(owner);self.mod.index(self.flat,self.nflat,self.lut,self.member)
 def run_indexed(self,owner,rounds=4,policy=1):
  assert owner.dtype==torch.int32 and owner.is_contiguous()
  self.mod.run_indexed(owner,self.ex,self.ow,self.res,self.lut,self.member,self.keys0,self.keys1,self.masks,self.phist,self.paccept,self.poffsets,self.temp,self.output,rounds,policy)
  return self.output
 def full_indexed(self,owner,rounds=4,policy=1):
  self.prepare_indexed(owner);return self.run_indexed(owner,rounds,policy)
