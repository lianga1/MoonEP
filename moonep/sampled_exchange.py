"""Source-local eight-rank bundle proposals and stable prefix quotas.

The native adapter supplies current all-gathered counts each round.
"""
from pathlib import Path
import hashlib
import torch
from torch.utils.cpp_extension import load

BUCKETS=8*8*2*9
_EXT=None

def extension():
    global _EXT
    if _EXT is None:
        p=Path(__file__).parent / 'csrc' / 'sampled_exchange.cu';sha=hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        _EXT=load(name='moonep_sampled_exchange_'+sha,sources=[str(p)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
    return _EXT

class SampledExchange:
    def __init__(self,routes,support,*,source,rounds=4):
        if routes.ndim!=2 or routes.shape[1]!=8 or routes.dtype!=torch.int32 or not routes.is_cuda or not routes.is_contiguous():
            raise ValueError('routes require contiguous CUDA int32 [S,8]')
        if tuple(support.shape)!=(8,352) or support.dtype!=torch.bool or not support.is_contiguous() or support.device!=routes.device:
            raise ValueError('support requires contiguous CUDA bool [8,352]')
        if source not in range(8) or rounds<1:raise ValueError('source 0..7 and positive rounds required')
        self.routes,self.support,self.source,self.rounds=routes,support,source,rounds
        self.S=routes.shape[0];self.blocks=(self.S+255)//256;self.ext=extension()
        def ints(*shape):return torch.empty(shape,device=routes.device,dtype=torch.int32)
        self.owner=torch.empty_like(routes);self.old_masks=torch.empty_like(routes);self.exec_masks=torch.empty_like(routes)
        self.resident=ints(352);self.keys=ints(self.S);self.moves=ints(self.S);self.ordinals=ints(self.S)
        self.block_counts=ints(BUCKETS,self.blocks);self.block_prefix=ints(BUCKETS,self.blocks)
        self.counts=ints(BUCKETS);self.total=ints(BUCKETS);self.source_prefix=ints(BUCKETS);self.accept=ints(BUCKETS)
        self.stats=torch.empty((rounds,4),device=routes.device,dtype=torch.int64)
    def prepare_support(self,support=None):
        self.ext.prepare(self.routes,self.support if support is None else support,self.resident,self.exec_masks)
    def reset(self,seed):self.ext.reset(seed,self.owner,self.old_masks)
    def propose(self,step):
        self.ext.propose(self.old_masks,self.exec_masks,self.keys,self.moves,self.ordinals,self.block_counts,self.block_prefix,self.counts,self.S,self.source*self.S,step)
        return self.counts
    def quota(self,step,gathered):self.ext.quota(gathered,self.total,self.source_prefix,self.accept,self.stats,self.source,step)
    def apply(self):
        self.ext.apply(self.keys,self.moves,self.ordinals,self.block_prefix,self.source_prefix,self.accept,self.old_masks,self.owner)
        return self.owner
    def commit(self,step,gathered):self.quota(step,gathered);return self.apply()
