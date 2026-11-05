"""Exact CUDA port of optimize_aggregate_multipair, eight logical ranks.
All state sorting, class counts, DP and expansion run on one physical GPU.
Includes per-round state grouping. Eager path; no claim of Graph capture.
"""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
_mod=None

def extension():
    global _mod
    if _mod is None:
        _mod=load(name='epmoe_aggregate_exact_cuda',sources=[str(Path(__file__).with_name('aggregate_cuda.cu'))],extra_cuda_cflags=['-O3','-lineinfo','--expt-relaxed-constexpr'],verbose=True)
    return _mod

class AggregateRunner:
    def __init__(self,routes,support,coalesced=True):
        self.routes=routes.to(dtype=torch.int32).contiguous()
        self.support=support.to(dtype=torch.bool).contiguous()
        self.mod=extension()
        self.coalesced=bool(coalesced)
    def run(self,owner,rounds=4):
        # Conversion excluded when caller already supplies contiguous int32.
        return self.mod.run(self.routes,owner.to(dtype=torch.int32).contiguous(),self.support,rounds,self.coalesced)[0]
