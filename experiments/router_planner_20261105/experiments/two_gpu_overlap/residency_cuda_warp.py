"""Experimental single-warp greedy variant; original implementation unchanged.

Only the greedy kernel's execution mapping changes. Methods, proposal order,
top-2 records and output/history buffers retain the verified base contract.
Must pass exact two-GPU comparison before any timing is accepted.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from torch.utils.cpp_extension import load
from residency_cuda import TwoRankResidency as BaseTwoRankResidency

_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        path = Path(__file__).with_suffix('.cu')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        _EXTENSION = load(name=f'epmoe_residency_r2_warp_{digest}', sources=[str(path)],
                          extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
                          verbose=os.environ.get('MOE_CUDA_BUILD_VERBOSE', '0') == '1')
    return _EXTENSION


class TwoRankResidency(BaseTwoRankResidency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext = extension()

    def source_manifest(self):
        here = Path(__file__)
        files = (here, here.with_suffix('.cu'), here.with_name('residency_cuda.py'), here.with_name('residency_cuda.cu'))
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
