"""Top-8 fixed-residency bucket exchange, single GPU or globally coordinated shards.

This is a component, not an end-to-end planner. Seed and residency preparation,
dispatch, combine, expert copying and gradient traffic are outside this class.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

BUCKETS = 8 * 8 * 2 * 9
_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        source = Path(__file__).with_suffix('.cu')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _EXTENSION = load(name=f'epmoe_bucket8_{digest}', sources=[str(source)],
                          extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
                          verbose=os.environ.get('MOE_CUDA_BUILD_VERBOSE', '0') == '1')
    return _EXTENSION


class BucketCudaPlanner:
    def __init__(self, routes, support, *, per_source, rounds=4,
                 global_token_offset=0, coordinated=False, process_group=None):
        if not (routes.is_cuda and routes.dtype == torch.int32 and routes.is_contiguous()
                and routes.ndim == 2 and routes.shape[1] == 8 and routes.shape[0] > 0):
            raise ValueError('routes require contiguous CUDA int32 [tokens,8]')
        if not (support.is_cuda and support.dtype == torch.bool and support.is_contiguous()
                and support.ndim == 2 and support.shape[0] == 8 and support.device == routes.device):
            raise ValueError('support requires contiguous CUDA bool [8,E] on routes device')
        self.routes = routes
        self.tokens = routes.shape[0]
        self.per_source = int(per_source)
        self.rounds = int(rounds)
        self.global_token_offset = int(global_token_offset)
        if self.rounds < 1 or self.per_source < 1:
            raise ValueError('rounds and per_source must be positive')
        if self.global_token_offset < 0 or self.global_token_offset + self.tokens > 8 * self.per_source:
            raise ValueError('tokens must belong to logical source ranks 0 through 7')
        self.coordinated = bool(coordinated)
        self.process_group = process_group
        if self.coordinated:
            if not dist.is_initialized():
                raise ValueError('coordinated mode requires initialized torch.distributed')
            self.rank = dist.get_rank(process_group)
            self.world = dist.get_world_size(process_group)
            if self.world != 8 or self.tokens != self.per_source or self.global_token_offset != self.rank * self.per_source:
                raise ValueError('coordinated mode requires eight source-ordered contiguous equal shards')
        else:
            self.rank, self.world = 0, 1
        self.token_bits = max(1, (self.tokens - 1).bit_length())
        if self.token_bits + 11 > 63:
            raise ValueError('token count exceeds sort-key capacity')
        self.ext = extension()
        device = routes.device
        self.owner = torch.empty_like(routes)
        self.resident = torch.empty(support.shape[1], dtype=torch.int32, device=device)
        self.ext.set_support(support, self.resident)
        self.keys = torch.empty(self.tokens, dtype=torch.int64, device=device)
        self.sorted = torch.empty_like(self.keys)
        self.move_masks = torch.empty(self.tokens, dtype=torch.int32, device=device)
        self.starts = torch.empty(BUCKETS + 1, dtype=torch.int32, device=device)
        self.ends = torch.empty_like(self.starts)
        self.accept = torch.zeros_like(self.starts)
        self.counts = torch.empty(BUCKETS, dtype=torch.int32, device=device)
        self.gathered = torch.empty(self.world * BUCKETS, dtype=torch.int32, device=device)
        self.total = torch.empty_like(self.counts)
        self.prefix = torch.empty_like(self.counts)
        self.stats = torch.empty((self.rounds, 4), dtype=torch.int64, device=device)
        nbytes = self.ext.workspace_size(self.keys, self.sorted, self.token_bits)
        self.workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)

    def reset(self, seed):
        if not (seed.device == self.routes.device and seed.dtype == torch.int32
                and seed.shape == self.owner.shape and seed.is_contiguous()):
            raise ValueError('seed must match routes shape/device and be contiguous int32')
        self.owner.copy_(seed)
        return self.owner

    def propose(self, step):
        self.ext.propose_round(self.routes, self.owner, self.resident, self.keys,
                               self.sorted, self.move_masks, self.starts, self.ends,
                               self.counts, self.workspace, self.token_bits,
                               self.per_source, self.global_token_offset, int(step))
        return self.counts

    def commit(self, step, gathered=None, rank=None, world=None):
        """Public split API also allows externally orchestrated count collection."""
        self.ext.finish_round(self.sorted, self.move_masks, self.starts,
                              self.counts if gathered is None else gathered,
                              self.total, self.prefix, self.accept, self.owner,
                              self.stats, self.token_bits,
                              0 if rank is None else int(rank),
                              1 if world is None else int(world), int(step))
        return self.owner

    def run_round(self, step):
        self.propose(step)
        if self.coordinated:
            # The only metadata collective per round: 1152 int32 per source.
            # Rank order equals source/token order, so prefix + local ordinal
            # equals the global stable-sort ordinal exactly.
            dist.all_gather_into_tensor(self.gathered, self.counts,
                                        group=self.process_group, async_op=False)
            self.commit(step, self.gathered, self.rank, self.world)
        else:
            self.commit(step)
        return self.owner

    def run(self, seed):
        self.reset(seed)
        for step in range(self.rounds):
            self.run_round(step)
        return self.owner

    def source_manifest(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), Path(__file__).with_suffix('.cu')]}
