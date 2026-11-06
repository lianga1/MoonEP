"""Bounded same-stream planner runtime; setup collectives are outside planning.

This is a same-node NVLink EP8 implementation, not a general NCCL replacement.
Only one planner invocation may be in flight per Buffer. All ranks must execute
the same collective sequence, including invalid-input/native-fallback paths.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


@lru_cache(None)
def extension():
    source=Path(__file__).parent/'csrc'/'sampled_runtime.cu'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(name=f'moonep_sampled_runtime_{digest}',sources=[str(source)],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)


class BoundedPlannerRuntime:
    def __init__(self,ctx,*,sm_limit,fuse,max_words):
        from .buffer import create_nvl_dist_tensor,pad_dim0_for_alignment
        self.rank,self.group=int(ctx['rank']),ctx.get('group')
        self.limit,self.fuse=int(sm_limit),bool(fuse)
        self.device=ctx['meta_buf'].device
        self.capacity=int(max_words)
        self.stride=pad_dim0_for_alignment([4+self.capacity],torch.int32)
        self.shared=create_nvl_dist_tensor([self.stride],torch.int32,self.rank,8,group=self.group)
        # Only the owner initializes its four header words. Setup is not timed.
        self.shared[self.rank*self.stride:self.rank*self.stride+4].zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.ext=extension()

    def _input(self,value):
        if not (value.is_cuda and value.device==self.device and value.dtype==torch.int32 and value.is_contiguous()):
            raise ValueError('bounded peer collectives require contiguous CUDA int32 on the Buffer device')
        if not 0<value.numel()<=self.capacity:
            raise ValueError('bounded peer collective exceeds its fixed workspace')

    def all_gather(self,src,dst):
        self._input(src)
        self.ext.collective(src,dst,self.shared,self.stride,self.rank,0,self.limit,self.fuse)
        return dst

    def all_reduce(self,value,op='sum'):
        self._input(value)
        if op not in ('sum','max'):
            raise ValueError('bounded planner reductions support sum/max only')
        self.ext.collective(value,value,self.shared,self.stride,self.rank,1 if op=='sum' else 2,self.limit,self.fuse)
        return value

    def copy(self,src,dst):
        self.ext.copy(src,dst,self.limit)

    def finish(self,adapter,candidate,candidate_cu,cu,plan):
        self.ext.status(adapter.residency.status,adapter.materializer.status,adapter.force_reject,adapter.status)
        self.all_reduce(adapter.status,'max')
        names=('dst','experts_to_copy','zero_fill_ranges','remote_stats')
        self.ext.select_all(adapter.status,adapter.valid,
            [candidate_cu]+[getattr(candidate,n) for n in names],
            [adapter.seed_cu]+[getattr(adapter.seed,n) for n in names],
            [cu]+[getattr(plan,n) for n in names],self.limit,self.fuse)
