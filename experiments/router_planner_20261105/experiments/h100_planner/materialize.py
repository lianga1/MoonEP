"""PyTorch GPU reference for owner -> executable slot/row/provenance materialization.
Fixed resident slots prepared outside run; stable per-slot ordering.
Includes destination dedup and padded offsets, excludes dispatch payloads.
"""
import torch
class Materializer:
 def __init__(self,routes,support,B=8,padding=128):
  self.routes=routes.reshape(-1).long();self.R=8;self.E=352;self.H=44;self.B=B;self.Q=44+B;self.T=routes.numel()//8;self.M=routes.numel();self.cap=self.M//8;self.NvS=self.cap+(padding-1)*88;self.padding=padding
  dev=routes.device;e=torch.arange(352,device=dev);r=torch.arange(8,device=dev)[:,None];extra=support&(e[None,:]//44!=r)
  extra_ord=extra.long().cumsum(1)-1
  self.lut=torch.where(e[None,:]//44==r,e[None,:]%44,torch.where(extra,44+extra_ord,-1)).long()
  self.flat_id=torch.arange(self.M,device=dev);self.tri=torch.tril(torch.ones((8,8),dtype=torch.bool,device=dev),diagonal=-1)
 def run(self,owner):
  o=owner.reshape(-1).long();slot=self.lut[o,self.routes];physical=o*self.Q+slot
  counts=torch.zeros(8*self.Q,device=o.device,dtype=torch.int64).scatter_add_(0,physical,torch.ones_like(physical)).reshape(8,self.Q)
  padded=((counts+self.padding-1)//self.padding)*self.padding
  ends=padded.cumsum(1);offsets=ends-padded
  zero_fill=torch.stack((torch.where(padded>counts,offsets+counts,0),padded-counts),dim=-1)
  order=torch.argsort(physical,stable=True)
  c=counts.reshape(-1);starts=c.cumsum(0)-c
  row=self.flat_id-starts[physical[order]]+offsets.reshape(-1)[physical[order]]
  raw=torch.empty_like(o);raw.scatter_(0,order,o[order]*self.NvS+row)
  provenance=torch.full((8*self.NvS,),-1,device=o.device,dtype=torch.int64)
  provenance.scatter_(0,raw,(self.flat_id//self.cap)*self.NvS+self.flat_id%self.cap)
  oo=o.reshape(-1,8);dup=((oo[:,:,None]==oo[:,None,:])&self.tri[None,:,:]).any(2)
  encoded=torch.where(dup,-raw.reshape(-1,8)-1,raw.reshape(-1,8))
  return dict(dst=encoded,physical_topk=physical,counts=counts,cu_seqlens=ends,zero_fill=zero_fill,provenance=provenance,raw_dst=raw)
