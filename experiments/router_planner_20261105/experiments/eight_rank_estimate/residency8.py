"""Full original-eight-rank sampled residency on one GPU, no CPU decisions.

All logical routes/owners and native copy lists already reside on this GPU.
run() includes runtime route/owner sampling and mandatory support rebuilding,
exact group frequencies/top2, package ordering, and one global budgeted greedy.
It excludes real eight-rank sample collection and support publication.
Unique top-8 expert IDs per token, d=256 sorted original global IDs, E=352.
The native [8,B] copy list must cover every non-home seed execution.
"""
from pathlib import Path
import hashlib
import os
import torch
from torch.utils.cpp_extension import load

_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        source = Path(__file__).with_suffix(".cu")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _EXT = load(name=f"epmoe_residency8_{digest}", sources=[str(source)],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=os.environ.get("MOE_CUDA_BUILD_VERBOSE", "0") == "1")
    return _EXT


class Residency8:
    def __init__(self, routes, sample_ids, *, per_source=16768, slots=8):
        if not (routes.is_cuda and routes.is_contiguous() and routes.dtype == torch.int32
                and routes.shape[-1] == 8 and routes.numel() == 8 * int(per_source) * 8):
            raise ValueError("routes must be contiguous CUDA int32 containing all eight logical source shards")
        if not (sample_ids.device == routes.device and sample_ids.dtype == torch.int64
                and sample_ids.is_contiguous() and sample_ids.shape == (256,)):
            raise ValueError("sample_ids must be sorted CUDA int64 [256] on the same device")
        if not 1 <= slots <= 8:
            raise ValueError("slots must be 1..8")
        self.routes = routes.reshape(-1, 8)
        self.ids = sample_ids
        self.per_source, self.slots = int(per_source), int(slots)
        self.ext = extension()
        device = routes.device
        def ints(*shape):
            return torch.empty(shape, dtype=torch.int32, device=device)
        def words(*shape):
            return torch.empty(shape, dtype=torch.int64, device=device)
        self.samples = ints(256, 16)
        self.support = torch.empty((8, 352), dtype=torch.bool, device=device)
        self.extra = ints(8)
        self.classes, self.frequencies = ints(28, 256), ints(28, 256)
        self.sig0, self.sig1, self.masks = words(28, 256), words(28, 256), words(28, 256, 6)
        self.tops, self.first = ints(28, 36, 2), ints(28, 36)
        self.records, self.sizes = ints(576, 4), ints(576)
        self.packages, self.package_count = ints(2688, 5), ints(1)
        self.experts_to_copy = ints(8, self.slots)
        self.history, self.history_count = words(8 * self.slots, 18), ints(1)

    def run(self, owner, native_experts_to_copy):
        owner = owner.reshape(-1, 8)
        self.ext.run(self.routes, owner, self.ids, native_experts_to_copy,
            self.samples, self.support, self.extra, self.classes, self.sig0, self.sig1,
            self.masks, self.frequencies, self.tops, self.first, self.records, self.sizes,
            self.packages, self.package_count, self.experts_to_copy, self.history,
            self.history_count, self.per_source, self.slots)
        return self.support

    def accepted_packages(self):
        """Synchronizing audit helper; never call inside benchmark timing."""
        count = int(self.history_count.item())
        rows = self.history[:count].cpu().tolist()
        result = []
        for row in rows:
            def experts(words):
                return [e for e in range(352) if (int(words[e // 64]) >> (e % 64)) & 1]
            result.append(dict(a=row[2], b=row[3], add_a=experts(row[6:12]),
                add_b=experts(row[12:18]), size=row[4], estimated_sample_gain=row[1]))
        return result

    def source_manifest(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_suffix(".cu"))}
