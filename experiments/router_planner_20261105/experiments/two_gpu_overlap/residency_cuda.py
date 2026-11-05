"""Exact sampled reciprocal residency for two ranks, entirely on the GPU.

Sample indices are fixed, sorted, global IDs prepared by the caller. Use
run(..., local_routes=routes) for full timing: route and owner samples are
gathered together through one collective. Omitting local_routes reuses cached
route samples and is a narrower component measurement. No CPU decisions/readback.
The mandatory table uses the native planner's validated experts_to_copy;
this replaces the NK scan only when that list covers every seed execution.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        path = Path(__file__).with_suffix('.cu')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        _EXTENSION = load(name=f'epmoe_residency_r2_{digest}', sources=[str(path)],
                          extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
                          verbose=os.environ.get('MOE_CUDA_BUILD_VERBOSE', '0') == '1')
    return _EXTENSION


class TwoRankResidency:
    def __init__(self, sample_routes, sample_global_ids, *, local_tokens, rank,
                 slots=8, process_group=None):
        if not (sample_routes.is_cuda and sample_routes.is_contiguous()
                and sample_routes.dtype == torch.int32 and sample_routes.ndim == 2
                and sample_routes.shape[1] == 8 and 1 <= sample_routes.shape[0] <= 1024):
            raise ValueError('sample_routes requires contiguous CUDA int32 [d,8], d<=1024')
        if not (sample_global_ids.device == sample_routes.device and sample_global_ids.is_contiguous()
                and sample_global_ids.dtype == torch.int64
                and sample_global_ids.shape == (sample_routes.shape[0],)):
            raise ValueError('sample_global_ids requires CUDA int64 [d] on the same device')
        if rank not in (0, 1) or not 1 <= slots <= 8 or local_tokens <= 0:
            raise ValueError('rank 0/1, 1<=slots<=8, local_tokens positive required')
        self.routes, self.ids = sample_routes.clone(), sample_global_ids
        self.local_tokens, self.rank, self.slots = int(local_tokens), int(rank), int(slots)
        self.process_group = process_group
        self.ext = extension()
        d = sample_routes.shape[0]
        device = sample_routes.device
        def ints(*shape):
            return torch.empty(shape, dtype=torch.int32, device=device)
        self.sample_owner = ints(d, 8)
        self.joint_samples = ints(d, 16)
        self.support = torch.empty((2, 352), dtype=torch.bool, device=device)
        self.extra = ints(2)
        self.classes, self.groups = ints(d), ints(d, 8)
        self.sig0 = torch.empty(d, dtype=torch.int64, device=device)
        self.sig1 = torch.empty_like(self.sig0)
        self.frequencies, self.tops, self.first = ints(d), ints(36, 2), ints(36)
        self.experts_to_copy = ints(2, slots)
        self.history = ints(2 * slots, 7)
        self.history_count = ints(1)
        self.diagnostics = ints(3)  # package count, final extra count at ranks 0/1

    def gather_local_owner(self, local_owner):
        """Fill this rank's sampled owner rows; non-owned rows contain -1."""
        if not (local_owner.shape == (self.local_tokens, 8)
                and local_owner.dtype == torch.int32 and local_owner.is_contiguous()
                and local_owner.device == self.routes.device):
            raise ValueError('local_owner must be contiguous CUDA int32 [S,8]')
        self.ext.gather_owner(local_owner, self.ids, self.sample_owner, self.local_tokens, self.rank)
        return self.sample_owner

    def gather_owner(self, local_owner):
        if not dist.is_initialized() or dist.get_world_size(self.process_group) != 2:
            raise RuntimeError('gather_owner requires a two-rank process group')
        if dist.get_rank(self.process_group) != self.rank:
            raise RuntimeError('process group ranks must match original source ranks')
        self.gather_local_owner(local_owner)
        dist.all_reduce(self.sample_owner, op=dist.ReduceOp.MAX, group=self.process_group)
        return self.sample_owner

    def gather_samples(self, local_routes, local_owner):
        """Refresh routes and owners from source shards using one collective."""
        for tensor in (local_routes, local_owner):
            if not (tensor.shape == (self.local_tokens, 8) and tensor.dtype == torch.int32
                    and tensor.is_contiguous() and tensor.device == self.routes.device):
                raise ValueError('local routes/owner need contiguous CUDA int32 [S,8]')
        if not dist.is_initialized() or dist.get_world_size(self.process_group) != 2:
            raise RuntimeError('gather_samples requires a two-rank process group')
        if dist.get_rank(self.process_group) != self.rank:
            raise RuntimeError('process-group rank must equal original source rank')
        self.ext.gather_joint(local_routes, local_owner, self.ids, self.joint_samples,
                              self.local_tokens, self.rank)
        dist.all_reduce(self.joint_samples, op=dist.ReduceOp.MAX, group=self.process_group)
        self.ext.unpack_joint(self.joint_samples, self.routes, self.sample_owner)
        return self.routes, self.sample_owner

    def select(self, native_experts_to_copy, global_sample_owner=None):
        """Select support from global sample owners and native mandatory copies.

        native_experts_to_copy must be the complete [2,B] matrix, with -1
        padding; validate it against the native plan before using this shortcut.
        history[:, :7] = (rank_a,rank_b,added_a,added_b,size,gain,cost).
        Read only history[:history_count] after completion.
        """
        owner = self.sample_owner if global_sample_owner is None else global_sample_owner
        if owner.shape != self.routes.shape or owner.device != self.routes.device or owner.dtype != torch.int32 or not owner.is_contiguous():
            raise ValueError('global_sample_owner must match sample_routes')
        self.ext.select(self.routes, self.ids, owner, native_experts_to_copy,
                        self.support, self.extra, self.classes, self.groups,
                        self.sig0, self.sig1, self.frequencies, self.tops,
                        self.first, self.experts_to_copy, self.history,
                        self.history_count, self.diagnostics, self.local_tokens, self.slots)
        return self.support

    def run(self, native_experts_to_copy, local_owner, local_routes=None):
        if local_routes is None:
            self.gather_owner(local_owner)
        else:
            self.gather_samples(local_routes, local_owner)
        return self.select(native_experts_to_copy)

    def source_manifest(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_suffix('.cu'))}
