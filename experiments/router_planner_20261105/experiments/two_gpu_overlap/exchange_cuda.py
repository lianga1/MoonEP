"""Two-rank canonical bundle exchange, deterministic across source shards.

The optimized path caches per-token executability, carries a compact owner
mask between rounds, and computes stable bucket ordinals without sorting.
Both paths implement the same selection and global quota as bucket_optimizer.
Preparation is explicit: call prepare_support() for every changed residency
matrix, and include that call when timing a changing-residency planner.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

BUCKETS = 72
_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        source = Path(__file__).with_suffix('.cu')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _EXTENSION = load(
            name=f'epmoe_two_rank_{digest}', sources=[str(source)],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
            verbose=os.environ.get('MOE_CUDA_BUILD_VERBOSE', '0') == '1')
    return _EXTENSION


class TwoRankExchange:
    def __init__(self, routes, support, *, rank, per_source=None, rounds=4,
                 variant='prefix', process_group=None):
        if not (routes.is_cuda and routes.dtype == torch.int32 and routes.is_contiguous()
                and routes.ndim == 2 and routes.shape[1] == 8 and routes.shape[0] > 0):
            raise ValueError('routes must be contiguous CUDA int32 [S,8]')
        if not (support.is_cuda and support.dtype == torch.bool and support.is_contiguous()
                and support.ndim == 2 and support.shape[0] == 2
                and support.device == routes.device):
            raise ValueError('support must be contiguous CUDA bool [2,E]')
        if variant not in ('prefix', 'sort') or rank not in (0, 1) or rounds < 1:
            raise ValueError('variant must be prefix/sort, rank 0/1, rounds positive')
        self.routes = routes
        self.tokens = routes.shape[0]
        self.per_source = self.tokens if per_source is None else int(per_source)
        if self.per_source != self.tokens:
            raise ValueError('each rank must hold exactly its complete source shard')
        self.rank, self.rounds = int(rank), int(rounds)
        self.variant, self.cached = variant, variant == 'prefix'
        self.process_group = process_group
        self.ext = extension()
        self.token_bits = max(1, (self.tokens - 1).bit_length())
        if self.token_bits + 7 > 63:
            raise ValueError('token count exceeds packed key capacity')
        device = routes.device
        def ints(*shape):
            return torch.empty(shape, dtype=torch.int32, device=device)
        self.owner = torch.empty_like(routes)
        self.resident = ints(support.shape[1])
        self.exec0, self.exec1, self.old0 = ints(self.tokens), ints(self.tokens), ints(self.tokens)
        self.moves, self.ordinals = ints(self.tokens), ints(self.tokens)
        self.keys = torch.empty(self.tokens, dtype=torch.int32 if self.cached else torch.int64, device=device)
        self.sorted = torch.empty(self.tokens, dtype=torch.int64, device=device)
        self.starts, self.ends = ints(BUCKETS + 1), ints(BUCKETS + 1)
        self.blocks = (self.tokens + 255) // 256
        self.block_counts, self.block_prefix = ints(BUCKETS, self.blocks), ints(BUCKETS, self.blocks)
        self.counts, self.source_prefix, self.accept = ints(BUCKETS), ints(BUCKETS), ints(BUCKETS)
        self.gathered = ints(2 * BUCKETS)
        self.stats = torch.empty((rounds, 4), dtype=torch.int64, device=device)
        nbytes = 0 if self.cached else self.ext.workspace_size(self.keys, self.sorted, self.token_bits)
        self.workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
        self.prepare_support(support)

    def prepare_support(self, support):
        """Refresh GPU caches after support changes; no host data readback."""
        if support.shape != (2, self.resident.numel()) or support.device != self.routes.device:
            raise ValueError('support shape/device changed')
        self.ext.prepare(self.routes, support, self.resident, self.exec0, self.exec1, self.cached)

    def reset(self, seed):
        if seed.device != self.routes.device or seed.shape != self.routes.shape:
            raise ValueError('seed must match routes shape/device')
        self.ext.reset(seed, self.owner, self.old0, self.cached)
        return self.owner

    def propose(self, step):
        if self.cached:
            self.ext.propose_prefix(
                self.old0, self.exec0, self.exec1, self.keys, self.moves,
                self.ordinals, self.block_counts, self.block_prefix, self.counts,
                self.rank, self.rank * self.per_source, int(step))
        else:
            self.ext.propose_sort(
                self.routes, self.owner, self.resident, self.keys, self.sorted,
                self.moves, self.starts, self.ends, self.counts, self.workspace,
                self.token_bits, self.rank, self.rank * self.per_source, int(step))
        return self.counts

    def commit(self, step, gathered):
        """gathered contains source 0 counts then source 1, on this device.

        stats[step] is GLOBAL (exchanges, moved pairs, accepted tokens, gain),
        identical on both ranks; do not sum it across ranks.
        """
        self.ext.commit(
            self.keys, self.sorted, self.moves, self.ordinals, self.starts,
            self.block_prefix, gathered, self.source_prefix, self.accept,
            self.old0, self.owner, self.stats, self.token_bits, self.rank,
            int(step), self.cached)
        return self.owner

    def run_round(self, step):
        if not dist.is_initialized() or dist.get_world_size(self.process_group) != 2:
            raise RuntimeError('run_round requires a two-rank process group; otherwise use propose/commit')
        if dist.get_rank(self.process_group) != self.rank:
            raise RuntimeError('process-group rank must equal original source rank')
        self.propose(step)
        dist.all_gather_into_tensor(self.gathered, self.counts, group=self.process_group)
        return self.commit(step, self.gathered)

    def run(self, seed, support=None):
        if support is not None:
            self.prepare_support(support)
        self.reset(seed)
        for step in range(self.rounds):
            self.run_round(step)
        return self.owner

    def source_manifest(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_suffix('.cu'))}
