"""Distributed sampled expert residency for the optional native planner.

R=8, E=352, K=8, global d=256, top-two groups per direction/size/gain.
Each source reads only its local routes/seed owners. One SUM collective joins
the sampled rows; all ranks then run the same global budgeted CUDA selector.
No routing rows, support tables, or frequencies are supplied from a trace.

Construct outside CUDA Graph capture. run() uses preallocated GPU storage and
performs no CPU data decisions or tensor-to-host reads. NCCL graph capture is
supported subject to the framework's normal collective warm-up requirements.
The caller must invoke this object in the same order on all eight group ranks.
The initial native copies must be the same complete global [8,B] plan on every
rank. They are mandatory, and all 44 home experts/rank are always retained.

status is a GPU int32[1] bitmask, reset on every run:
  1 malformed native copy list; 2 malformed sampled route/owner or duplicate
  top-k expert; 4 missing/duplicated source sample; 8 sampled seed execution
  absent from native support; 16 mandatory budget overflow.
Nonzero status requires the caller's collective native fallback. A zero status
does NOT certify unsampled seed pairs: the full materializer must independently
check all pairs, physical slots and budgets and participate in global fallback.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


BAD_COPIES = 1
BAD_SAMPLE = 2
BAD_OWNERSHIP = 4
UNSUPPORTED_SEED = 8
BUDGET_OVERFLOW = 16
_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        source = Path(__file__).parent / "csrc" / "sampled_residency.cu"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _EXTENSION = load(name=f"moonep_sampled_residency_{digest}", sources=[str(source)],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=os.environ.get("MOONEP_CUDA_BUILD_VERBOSE", "0") == "1")
    return _EXTENSION


class Residency:
    """Runtime source-sharded selector; result tensors are borrowed storage.

    ctx requires R/E/K/S/B/rank/device and optional group. sample_seed is an
    initialization-time integer and must match across the process group. B may
    be 8 or the native default 44 (all values 1..44 are accepted).

    Call run(local_routes, local_owner, native_copies) on the intended CUDA
    stream. There is one all_reduce of 4,609 int32 values (18,436 bytes): sampled
    experts, sampled owners, row ownership/validity, and copy-list validity.
    The resident/expert/slot decisions remain global across all 28 rank pairs.
    """

    def __init__(self, ctx, sample_seed=20261026, *, component_only=False):
        self.ranks, self.experts, self.topk = int(ctx["R"]), int(ctx["E"]), int(ctx["K"])
        self.per_source, self.slots, self.rank = int(ctx["S"]), int(ctx["B"]), int(ctx["rank"])
        self.group = ctx.get("group")
        self.component_only = bool(component_only)
        if (self.ranks, self.experts, self.topk) != (8, 352, 8):
            raise ValueError("sampled residency requires R=8, E=352, K=8")
        if not 1 <= self.slots <= 44 or self.per_source < 32:
            raise ValueError("sampled residency requires 1<=B<=44 and global N>=256")
        if not 0 <= self.rank < 8:
            raise ValueError("ctx rank must be an original logical rank in 0..7")
        if not self.component_only:
            if not dist.is_initialized() or dist.get_world_size(self.group) != 8:
                raise ValueError("sampled residency requires an initialized eight-rank process group")
            if dist.get_rank(self.group) != self.rank:
                raise ValueError("ctx rank must be the process-group rank")
            if str(dist.get_backend(self.group)).lower() != "nccl":
                raise ValueError("CUDA sampled residency requires an NCCL process group")
        raw_device = ctx["device"]
        self.device = torch.device("cuda", raw_device) if isinstance(raw_device, int) else torch.device(raw_device)
        if self.device.type != "cuda":
            raise ValueError("sampled residency requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.sample_seed = int(sample_seed)
        # Initialization only. No token contents are observed by this RNG.
        sample_ids = np.sort(np.random.default_rng(self.sample_seed).choice(
            8 * self.per_source, 256, replace=False)).astype(np.int64)
        self.ids = torch.as_tensor(sample_ids, device=self.device)
        self.ext = extension()
        def ints(*shape):
            return torch.empty(shape, dtype=torch.int32, device=self.device)
        def words(*shape):
            return torch.empty(shape, dtype=torch.int64, device=self.device)
        self.packet = ints(256 * 18 + 1)
        self.status = ints(1)
        self.samples = ints(256, 16)
        self.support = torch.empty((8, 352), dtype=torch.bool, device=self.device)
        self.extra = ints(8)
        self.classes, self.frequencies = ints(28, 256), ints(28, 256)
        self.sig0, self.sig1, self.masks = words(28, 256), words(28, 256), words(28, 256, 6)
        self.tops, self.first = ints(28, 36, 2), ints(28, 36)
        self.records, self.sizes = ints(576, 4), ints(576)
        self.packages, self.package_count = ints(2688, 5), ints(1)
        self.experts_to_copy = ints(8, self.slots)
        self.history, self.history_count = words(8 * self.slots, 18), ints(1)

    def _input(self, value, shape, name):
        if not (value.device == self.device and value.dtype == torch.int32
                and value.is_contiguous() and tuple(value.shape) == shape):
            raise ValueError(f"{name} requires contiguous CUDA int32 {shape} on {self.device}")

    def gather_local(self, local_routes, local_owner, native_copies, *, source_rank=None):
        """Write one original source's packet and reset status; no collective.

        source_rank override is only for explicit component_only objects. A
        harness may sum four original sources on each of two physical GPUs,
        then SUM those two packets before build_from_reduced_samples(). That
        exercise measures components and is not eight-physical-rank execution.
        """
        self._input(local_routes, (self.per_source, 8), "local_routes")
        self._input(local_owner, (self.per_source, 8), "local_owner")
        self._input(native_copies, (8, self.slots), "native_copies")
        source = self.rank if source_rank is None else int(source_rank)
        if not 0 <= source < 8 or (source != self.rank and not self.component_only):
            raise ValueError("source override requires component_only=True and rank in 0..7")
        self.ext.gather(local_routes, local_owner, self.ids, native_copies,
            self.packet, self.status, self.per_source, source, self.slots)
        return self.packet

    def select(self, native_copies):
        """Consume the globally summed packet; run() is the normal entry point."""
        self._input(native_copies, (8, self.slots), "native_copies")
        self.ext.select(self.packet, self.ids, native_copies, self.samples,
            self.support, self.extra, self.classes, self.sig0, self.sig1, self.masks,
            self.frequencies, self.tops, self.first, self.records, self.sizes,
            self.packages, self.package_count, self.experts_to_copy, self.history,
            self.history_count, self.status, self.per_source, self.slots)
        return self.support

    def run(self, local_routes, local_owner, native_copies):
        if self.component_only:
            raise RuntimeError("component_only objects cannot run the native eight-rank collective path")
        self.gather_local(local_routes, local_owner, native_copies)
        dist.all_reduce(self.packet, op=dist.ReduceOp.SUM, group=self.group, async_op=False)
        return self.select(native_copies)

    def build_from_reduced_samples(self, native_copies, reduced_packet=None):
        """Select after an externally orchestrated SUM of original-source packets.

        If reduced_packet is supplied, copy it into borrowed packet storage on
        the current CUDA stream. No host reads or input-dependent allocations.
        All source contributions must use this object's same sample indices.
        """
        if reduced_packet is not None:
            self._input(reduced_packet, (256 * 18 + 1,), "reduced_packet")
            self.packet.copy_(reduced_packet)
        # A component harness may skip gather_local on this object; reset the
        # previous run's status before validating the complete current packet.
        self.status.zero_()
        return self.select(native_copies)

    def accepted_packages(self):
        """Explicit synchronizing debug readback; never used by run()."""
        count = int(self.history_count.item())
        rows = self.history[:count].cpu().tolist()
        result = []
        for row in rows:
            def unpack(words):
                return [e for e in range(352) if (int(words[e // 64]) >> (e % 64)) & 1]
            result.append(dict(a=row[2], b=row[3], add_a=unpack(row[6:12]),
                add_b=unpack(row[12:18]), size=row[4], estimated_sample_gain=row[1]))
        return result

    def source_manifest(self):
        paths = (Path(__file__), Path(__file__).parent / "csrc" / "sampled_residency.cu")
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
