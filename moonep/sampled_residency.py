"""Distributed sampled expert residency for the optional native planner.

R=8; expert count, top-k, global sample size and retained groups are configurable.
Each source reads only its local routes/seed owners. One SUM collective joins
the sampled rows; all ranks then run the same global budgeted CUDA selector.
No routing rows, support tables, or frequencies are supplied from a trace.

Construct outside CUDA Graph capture. run() uses preallocated GPU storage and
performs no CPU data decisions or tensor-to-host reads. NCCL graph capture is
supported subject to the framework's normal collective warm-up requirements.
The caller must invoke this object in the same order on all eight group ranks.
The initial native copies must be the same complete global [8,B] plan on every
rank. They are mandatory, and all E/R home experts/rank are always retained.

status is a GPU int32[1] bitmask, reset on every run:
  1 malformed native copy list; 2 malformed sampled route/owner or duplicate
  top-k expert; 4 missing/duplicated source sample; 8 sampled seed execution
  absent from native support; 16 mandatory budget overflow; 32 internal
  package-capacity violation.
Nonzero status requires the caller's collective native fallback. A zero status
does NOT certify unsampled seed pairs: the full materializer must independently
check all pairs, physical slots and budgets and participate in global fallback.
"""
from __future__ import annotations

import hashlib
import os
import operator
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
INTERNAL_ERROR = 32
_EXTENSIONS = {}


def extension(experts, topk, samples, groups):
    parameters = (int(experts), int(topk), int(samples), int(groups))
    source = Path(__file__).parent / "csrc" / "sampled_residency.cu"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    key = (*parameters, digest)
    if key not in _EXTENSIONS:
        suffix = "_".join(map(str, parameters))
        defines = [f"-DMOONEP_{name}={value}" for name, value in zip(("E", "K", "D", "G"), parameters)]
        _EXTENSIONS[key] = load(name=f"moonep_sampled_residency_{suffix}_{digest}", sources=[str(source)],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo", *defines],
            verbose=os.environ.get("MOONEP_CUDA_BUILD_VERBOSE", "0") == "1")
    return _EXTENSIONS[key]


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


class Residency:
    """Runtime source-sharded selector; result tensors are borrowed storage.

    ctx requires R/E/K/S/B/rank/device and optional group. sample_seed is an
    initialization-time integer and must match across the process group.
    E is 8..4096 and divisible by 8; K is 1..min(32,E); B is 1..E-E/8.
    sample_size is 1..4096 and is capped at the actual global token count.
    groups_per_class is 1..8; the default 2 preserves the previous algorithm.
    candidate_limit=0 preserves unlimited selection. A positive limit retains
    that many complete reciprocal packages, ranked at mandatory support by
    gain/cost, gain, lower cost, then original package ID. The later greedy
    loop recomputes the retained packages' current scores. Limits above the
    model-specific package capacity are clamped to that capacity. All settings
    and storage are fixed at construction, before CUDA Graph capture.

    Call run(local_routes, local_owner, native_copies) on the intended CUDA
    stream. There is one all_reduce of D*(2*K+2)+1 int32 values: sampled
    experts, sampled owners, row ownership/validity, and copy-list validity.
    The resident/expert/slot decisions remain global across all 28 rank pairs.
    """

    def __init__(self, ctx, sample_seed=20261026, *, sample_size=256,
                 groups_per_class=2, candidate_limit=0, component_only=False):
        self.ranks, self.experts, self.topk = (_integer(ctx[key], key) for key in ("R", "E", "K"))
        self.per_source, self.slots, self.rank = (_integer(ctx[key], key) for key in ("S", "B", "rank"))
        self.group = ctx.get("group")
        self.component_only = bool(component_only)
        if self.ranks != 8 or not 8 <= self.experts <= 4096 or self.experts % 8:
            raise ValueError("sampled residency requires R=8 and E in 8..4096 divisible by 8")
        if not 1 <= self.topk <= min(32, self.experts):
            raise ValueError("sampled residency requires 1<=K<=min(32,E)")
        if not 1 <= self.slots <= self.experts - self.experts // 8 or self.per_source < 1:
            raise ValueError("sampled residency requires 1<=B<=E-E/8 and S>=1")
        self.requested_sample_size = _integer(sample_size, "sample_size")
        self.groups_per_class = _integer(groups_per_class, "groups_per_class")
        if not 1 <= self.requested_sample_size <= 4096 or not 1 <= self.groups_per_class <= 8:
            raise ValueError("sample_size must be 1..4096 and groups_per_class must be 1..8")
        self.sample_size = min(self.requested_sample_size, 8 * self.per_source)
        self.words = (self.experts + 63) // 64
        self.packet_width = 2 * self.topk + 2
        self.packet_size = self.sample_size * self.packet_width + 1
        self.classes_per_pair = 4 * (self.topk + 1)
        self.buckets = 64 * (self.topk + 1)
        self.max_packages = 28 * self.topk * 3 * self.groups_per_class ** 2
        self.candidate_limit = _integer(candidate_limit, "candidate_limit")
        if not 0 <= self.candidate_limit <= 2**31-1:
            raise ValueError("candidate_limit must be in 0..2147483647; 0 is unlimited")
        self.effective_candidate_limit = min(self.candidate_limit, self.max_packages)
        self.history_width = 6 + 2 * self.words
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
        self.sample_seed = _integer(sample_seed, "sample_seed")
        # Initialization only. No token contents are observed by this RNG.
        sample_ids = np.sort(np.random.default_rng(self.sample_seed).choice(
            8 * self.per_source, self.sample_size, replace=False)).astype(np.int64)
        self.ids = torch.as_tensor(sample_ids, device=self.device)
        self.ext = extension(self.experts, self.topk, self.sample_size, self.groups_per_class)
        self.uses_active_table = bool(self.ext.uses_active_table())
        def ints(*shape):
            return torch.empty(shape, dtype=torch.int32, device=self.device)
        def words(*shape):
            return torch.empty(shape, dtype=torch.int64, device=self.device)
        self.packet = ints(self.packet_size)
        self.status = ints(1)
        self.samples = ints(self.sample_size, 2 * self.topk)
        self.support = torch.empty((8, self.experts), dtype=torch.bool, device=self.device)
        self.copy_seen = ints(8, self.experts)
        self.extra = ints(8)
        self.classes, self.frequencies = ints(28, self.sample_size), ints(28, self.sample_size)
        self.groups = ints(28, self.sample_size, self.topk)
        # Hashes only accelerate rejection. Equal hashes always trigger exact
        # comparison of sorted expert IDs; no hash determines a tie or merge.
        self.sig0, self.sig1 = words(28, self.sample_size), words(28, self.sample_size)
        self.masks = words(28, self.sample_size, self.words)
        self.tops = ints(28, self.classes_per_pair, self.groups_per_class)
        self.first = ints(28, self.classes_per_pair)
        self.records, self.sizes = ints(self.buckets, 2 * self.groups_per_class), ints(self.buckets)
        self.package_offsets = ints(28 * self.topk + 1)
        self.packages, self.package_count = ints(self.max_packages, 5), ints(1)
        self.experts_to_copy = ints(8, self.slots)
        self.history, self.history_count = words(8 * self.slots, self.history_width), ints(1)
        # Unlimited selection retains its old execution path and allocates no
        # pruning storage. The CUB temporary buffer is queried once here, never
        # during run()/select() or a CUDA Graph replay.
        if self.effective_candidate_limit:
            self.scores_in, self.scores_out = ints(self.max_packages, 3), ints(self.max_packages, 3)
            needed = self.ext.pruning_workspace_bytes(self.scores_in, self.scores_out)
            self.sort_scratch = torch.empty(max(1, needed), dtype=torch.uint8, device=self.device)
            self.selected_ids, self.retained_count = ints(self.max_packages), ints(1)
            self.pruning_diagnostics = ints(3)
            self.active_keys = ints(self.buckets)
            self.active_rows = ints(self.buckets * 2 * self.groups_per_class)
            self.active_flags = ints(self.buckets + self.active_rows.numel())
            self.active_counts = torch.zeros(2, dtype=torch.int32, device=self.device)

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
        self._input(local_routes, (self.per_source, self.topk), "local_routes")
        self._input(local_owner, (self.per_source, self.topk), "local_owner")
        self._input(native_copies, (8, self.slots), "native_copies")
        source = self.rank if source_rank is None else int(source_rank)
        if not 0 <= source < 8 or (source != self.rank and not self.component_only):
            raise ValueError("source override requires component_only=True and rank in 0..7")
        self.ext.gather(local_routes, local_owner, self.ids, native_copies,
            self.packet, self.status, self.copy_seen, self.per_source, source, self.slots)
        return self.packet

    def select(self, native_copies):
        """Consume the globally summed packet; run() is the normal entry point."""
        self._input(native_copies, (8, self.slots), "native_copies")
        if self.effective_candidate_limit:
            self.ext.select_capped_active(self.packet, self.ids, native_copies, self.samples,
                self.support, self.extra, self.copy_seen, self.classes, self.groups, self.sig0, self.sig1, self.masks,
                self.frequencies, self.tops, self.first, self.records, self.sizes,
                self.package_offsets, self.packages, self.package_count, self.experts_to_copy, self.history,
                self.history_count, self.status, self.per_source, self.slots,
                self.scores_in, self.scores_out, self.sort_scratch, self.selected_ids,
                self.retained_count, self.pruning_diagnostics, self.effective_candidate_limit,
                self.active_keys, self.active_rows, self.active_flags, self.active_counts)
            return self.support
        self.ext.select(self.packet, self.ids, native_copies, self.samples,
            self.support, self.extra, self.copy_seen, self.classes, self.groups, self.sig0, self.sig1, self.masks,
            self.frequencies, self.tops, self.first, self.records, self.sizes,
            self.package_offsets, self.packages, self.package_count, self.experts_to_copy, self.history,
            self.history_count, self.status, self.per_source, self.slots)
        return self.support

    def run(self, local_routes, local_owner, native_copies):
        if self.component_only:
            raise RuntimeError("component_only objects cannot run the native eight-rank collective path")
        self.gather_local(local_routes, local_owner, native_copies)
        dist.all_reduce(self.packet, op=dist.ReduceOp.SUM, group=self.group, async_op=False)
        return self.select(native_copies)

    def candidate_diagnostics(self):
        """Synchronizing debug readback; never called by the planning path.

        active_keys/rows are cache work-list sizes, not expert counts. They
        are None for unlimited selection or the large-configuration generic
        kernel, which avoids the shared-memory cache specialization.
        """
        full = int(self.package_count.item())
        if self.effective_candidate_limit:
            eligible, before, retained = self.pruning_diagnostics.cpu().tolist()
            if before != full or retained != min(self.effective_candidate_limit, full):
                raise RuntimeError("candidate pruning diagnostics are inconsistent")
            selected = self.selected_ids[:retained].cpu().tolist()
            if len(set(selected)) != retained or any(i < 0 or i >= full for i in selected):
                raise RuntimeError("candidate pruning produced invalid package IDs")
        else:
            eligible, retained, selected = None, full, list(range(full))
        active = (self.active_counts.cpu().tolist()
                  if self.effective_candidate_limit and self.uses_active_table else [None, None])
        return dict(candidate_limit=self.candidate_limit,
            effective_candidate_limit=self.effective_candidate_limit,
            full_P=full, retained_P=retained, initial_eligible_P=eligible,
            history_I=int(self.history_count.item()), selected_ids=selected,
            active_keys=active[0], active_rows=active[1], uses_active_table=self.uses_active_table)

    def build_from_reduced_samples(self, native_copies, reduced_packet=None):
        """Select after an externally orchestrated SUM of original-source packets.

        If reduced_packet is supplied, copy it into borrowed packet storage on
        the current CUDA stream. No host reads or input-dependent allocations.
        All source contributions must use this object's same sample indices.
        """
        if reduced_packet is not None:
            self._input(reduced_packet, (self.packet_size,), "reduced_packet")
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
                return [e for e in range(self.experts) if (int(words[e // 64]) >> (e % 64)) & 1]
            result.append(dict(a=row[2], b=row[3], add_a=unpack(row[6:6 + self.words]),
                add_b=unpack(row[6 + self.words:6 + 2 * self.words]), size=row[4], estimated_sample_gain=row[1]))
        return result

    def source_manifest(self):
        paths = (Path(__file__), Path(__file__).parent / "csrc" / "sampled_residency.cu")
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
