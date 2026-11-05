"""Independent optimized residency variants; verified base source is unchanged.

Variants: tops = class-parallel top-2 reduction; warp = one-warp greedy;
combined = both; control = original kernels compiled in this translation unit.
All run/gather/select semantics and buffers are inherited from the base class.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from torch.utils.cpp_extension import load
from residency_cuda import TwoRankResidency as BaseTwoRankResidency

_EXTENSION = None
FLAGS = dict(control=0, tops=1, warp=2, combined=3)


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        path = Path(__file__).with_suffix('.cu')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        _EXTENSION = load(name=f'epmoe_residency_r2_optimized_{digest}', sources=[str(path)],
                          extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
                          verbose=os.environ.get('MOE_CUDA_BUILD_VERBOSE', '0') == '1')
    return _EXTENSION


class TwoRankResidency(BaseTwoRankResidency):
    def __init__(self, *args, variant='combined', **kwargs):
        if variant not in FLAGS:
            raise ValueError(f'variant must be one of {tuple(FLAGS)}')
        super().__init__(*args, **kwargs)
        self.variant, self.flags = variant, FLAGS[variant]
        self.ext = extension()

    def select(self, native_experts_to_copy, global_sample_owner=None):
        owner = self.sample_owner if global_sample_owner is None else global_sample_owner
        if owner.shape != self.routes.shape or owner.device != self.routes.device or owner.dtype != self.routes.dtype or not owner.is_contiguous():
            raise ValueError('global_sample_owner must match sample_routes')
        self.ext.select(self.routes, self.ids, owner, native_experts_to_copy,
                        self.support, self.extra, self.classes, self.groups,
                        self.sig0, self.sig1, self.frequencies, self.tops,
                        self.first, self.experts_to_copy, self.history,
                        self.history_count, self.diagnostics, self.local_tokens,
                        self.slots, self.flags)
        return self.support

    def source_manifest(self):
        here = Path(__file__)
        files = (here, here.with_suffix('.cu'), here.with_name('residency_cuda.py'), here.with_name('residency_cuda.cu'))
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
