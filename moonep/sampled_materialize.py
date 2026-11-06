"""Materialize sampled-group owners into this parent version's MoonEP plan.

Only this module's candidate buffers are written before global acceptance.
The caller combines all planner status tensors, reduces that decision across
the EP group, selects candidate or native seed into the final plan, and then
calls publish(final_plan.dst). Fresh native dispatch builds dup_groups,
dup_loffs and dup_counts; this module deliberately does not build them.

All methods enqueue on the current CUDA stream. Allocate/JIT before capture.
The caller must use identical support on every EP rank and invoke collectives
in the same order. Buffers are single-flight and borrowed: preserve the final
plan/dedup state and weight-slot version through backward before reusing them.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


@lru_cache(None)
def extension():
    source = Path(__file__).parent / "csrc" / "sampled_materialize.cu"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(
        name=f"moonep_sampled_materialize_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


class SampledMaterializer:
    """Local source tasks -> globally unique native destination rows.

    Required order: native seed -> prepare_support -> materialize -> caller's
    global status decision/selection -> publish. prepare_support resets status;
    materialize only accumulates errors. A nonzero status means the candidate
    must not be published. There is no host read or implicit status collective.

    The default uses native C1's stable expert order and stably sorts only the
    new execution rank. This keeps (rank, expert, original source offset)
    order without re-sorting expert bits. Set reuse_native_order=False for a
    general stable composite-key baseline. Both assign identical rows.

    sm_limit>0 replaces the device-wide sort with stable eight-bucket grouping
    and bounds every local launch, including initialization/publication, to
    that many CTAs. It requires native order reuse and an all_gather(src,dst)
    collective obeying the same budget. fuse combines metadata, validation and
    tile histogram generation; it does not change grouping or tie order.
    """

    STATUS = {
        1: "invalid support or missing home expert",
        2: "support needs more than B remote slots",
        4: "native seed execution is not covered by physical support",
        8: "invalid route ID or duplicate experts within a token",
        16: "invalid final owner or unsupported final execution",
        32: "destination task count differs from S*K",
        64: "padded rows exceed NvS or encoded row is invalid",
        128: "native C1 order is invalid or not expert-sorted",
    }

    def __init__(self, ctx, *, reuse_native_order=True, sm_limit=0, fuse=False,
                 collective=None):
        from moonep.planning import allocate_planning_outputs

        self.ctx, self.group = ctx, ctx.get("group")
        self.R, self.E, self.K, self.B, self.S, self.rank, self.NvS = (
            int(ctx[k]) for k in ("R", "E", "K", "B", "S", "rank", "NvS")
        )
        if not (self.R == 8 and 8 <= self.E <= 4096 and self.E % self.R == 0
                and 1 <= self.K <= min(32, self.E)
                and 1 <= self.B <= self.E - self.E // self.R):
            raise ValueError("sampled materializer requires R=8, 8<=E<=4096, E%8=0, "
                             "1<=K<=min(32,E), 1<=B<=E-E/8")
        self.N, self.padding = self.S * self.K, int(ctx["token_padding"])
        if self.S <= 0 or self.padding < 1 or self.NvS < self.N:
            raise ValueError("invalid local token count, padding or NvS")
        if self.R * self.NvS > 2**31 - 1:
            raise ValueError("encoded native destination addresses exceed int32")
        self.device = ctx["meta_buf"].device
        self.ext = extension()
        self.reuse_native_order = bool(reuse_native_order)
        if isinstance(sm_limit, bool) or not isinstance(sm_limit, int) or sm_limit < 0:
            raise ValueError("sm_limit must be a nonnegative integer")
        self.sm_limit, self.fuse, self.collective = sm_limit, bool(fuse), collective
        if self.sm_limit and not self.reuse_native_order:
            raise ValueError("bounded materialization requires reuse_native_order=True")
        if self.sm_limit and collective is None:
            raise ValueError("bounded materialization requires a bounded collective")
        self.plan, self.cu = allocate_planning_outputs(ctx)
        self.copies = self.plan.experts_to_copy

        def ints(*shape):
            return torch.empty(shape, dtype=torch.int32, device=self.device)

        self.status = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.lut = ints(self.R, self.E)
        self.counts = ints(self.R, self.E)
        self.gathered_counts = ints(self.R, self.R, self.E)
        self.offsets, self.prefix, self.totals = (ints(self.R, self.E) for _ in range(3))
        self.keys_in, self.keys_out, self.ids_in, self.ids_out = (ints(self.N) for _ in range(4))
        # One byte per pair, generated once in original token order using
        # subwarp match/ballot operations; avoids per-pair previous-K rescans.
        self.duplicates = torch.empty((self.S, self.K), dtype=torch.uint8, device=self.device)
        self.starts = ints(self.R, self.E)
        begin = self.rank * int(ctx["meta_chunk_padded"]) + int(ctx["ORDER_OFF"])
        self.native_order = ctx["meta_buf"][begin : begin + self.N]
        self.sort_bits = (self.R - 1).bit_length() if self.reuse_native_order else (self.R * self.E - 1).bit_length()
        if self.sm_limit:
            # Stable eight-bucket grouping over virtual 256-pair tiles. The
            # physical CTA budget never changes the order of these tiles.
            tiles = (self.N + 255) // 256
            self.sort_counts, self.sort_prefix = ints(self.R, tiles), ints(self.R, tiles)
            self.workspace = None
        else:
            workspace = self.ext.workspace(self.keys_in, self.keys_out, self.ids_in, self.ids_out, self.sort_bits)
            self.workspace = torch.empty(max(1, workspace), dtype=torch.uint8, device=self.device)

    def _int(self, tensor, shape, name):
        if (tensor.device != self.device or tensor.dtype != torch.int32
                or tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous CUDA int32 {tuple(shape)} on {self.device}")

    def decode_owner(self, raw_dst, out_owner):
        """Decode this parent's canonical duplicate marker: raw = -dst - 1."""
        self._int(raw_dst, (self.N,), "raw_dst")
        self._int(out_owner, (self.S, self.K), "out_owner")
        if self.sm_limit:
            self.ext.decode_bounded(raw_dst, out_owner, self.NvS, self.sm_limit)
        else:
            self.ext.decode(raw_dst, out_owner, self.NvS)
        return out_owner

    def prepare_support(self, support):
        """Build stable ascending-expert copy slots; reset this round's status.

        Support must already be identical on all ranks. This performs no EP
        collective and changes neither native metadata nor the seed plan.
        """
        if (support.device != self.device or support.dtype not in (torch.bool, torch.int32)
                or tuple(support.shape) != (self.R, self.E) or not support.is_contiguous()):
            raise ValueError(f"support must be contiguous CUDA bool/int32 [{self.R},{self.E}]")
        if self.sm_limit:
            self.ext.support_bounded(support, self.lut, self.copies, self.status,
                                     self.R, self.E, self.B, self.sm_limit)
        else:
            self.ext.support(support, self.lut, self.copies, self.status, self.R, self.E, self.B)
        return self.copies

    def materialize(self, routes, owner, seed_owner):
        """Build a candidate; caller still must globally accept it before use.

        Includes one all-gather of [destination,expert] local counts. All tasks
        are assigned a unique row by (destination,expert,source,local order).
        Invalid inputs still follow the same collective sequence and leave a
        rejected candidate without touching shared native metadata.
        """
        for name, tensor in (("routes", routes), ("owner", owner), ("seed_owner", seed_owner)):
            self._int(tensor, (self.S, self.K), name)
        if self.sm_limit:
            self.ext.prepare_bounded(
                routes, owner, seed_owner, self.native_order, self.lut,
                self.keys_in, self.keys_out, self.ids_in, self.ids_out,
                self.starts, self.counts, self.sort_counts, self.sort_prefix,
                self.duplicates, self.status, self.R, self.E, self.K,
                self.sm_limit, self.fuse,
            )
            self.collective.all_gather(self.counts.view(-1), self.gathered_counts.view(-1))
            self.ext.layout_encode_bounded(
                routes, owner, self.ids_out, self.starts, self.gathered_counts,
                self.lut, self.copies, self.offsets, self.prefix, self.totals, self.duplicates,
                self.plan.dst, self.cu, self.plan.zero_fill_ranges, self.plan.remote_stats,
                self.status, self.R, self.E, self.B, self.K, self.rank, self.padding,
                self.NvS, self.sm_limit,
            )
        else:
            self.ext.prepare(
                routes, owner, seed_owner, self.native_order, self.lut,
                self.keys_in, self.keys_out, self.ids_in, self.ids_out,
                self.starts, self.counts, self.workspace, self.duplicates, self.status,
                self.R, self.E, self.K, self.reuse_native_order, self.sort_bits,
            )
            dist.all_gather_into_tensor(self.gathered_counts.view(-1), self.counts.view(-1), group=self.group)
            self.ext.layout_encode(
                routes, owner, self.ids_out, self.starts, self.gathered_counts,
                self.lut, self.copies, self.offsets, self.prefix, self.totals, self.duplicates,
                self.plan.dst, self.cu, self.plan.zero_fill_ranges, self.plan.remote_stats,
                self.status, self.R, self.E, self.B, self.K, self.rank, self.padding, self.NvS,
            )
        return self.plan, self.cu

    def publish(self, final_dst):
        """Publish only the caller's globally selected native-compatible plan.

        Each rank clears its own src_info, all ranks synchronize, each source
        writes its selected pair provenance through the peer-mapped meta_buf,
        then all ranks synchronize again. This is also required after fallback:
        seed dst is selected verbatim, while scratch provenance is republished.
        This method cannot detect a caller that publishes a rejected candidate.
        """
        from moonep.inter_rank_sync import launch_inter_rank_sync

        self._int(final_dst, (self.N,), "final_dst")
        stride, offset = int(self.ctx["meta_chunk_padded"]), int(self.ctx["SRC_INFO_OFF"])
        begin = self.rank * stride + offset
        if self.sm_limit:
            self.ext.fill_bounded(self.ctx["meta_buf"], begin, self.NvS, -1, self.sm_limit)
        else:
            self.ctx["meta_buf"][begin : begin + self.NvS].fill_(-1)
        launch_inter_rank_sync(self.ctx)
        if self.sm_limit:
            self.ext.publish_bounded(final_dst, self.ctx["meta_buf"], self.rank,
                                     self.NvS, stride, offset, self.sm_limit)
        else:
            self.ext.publish(final_dst, self.ctx["meta_buf"], self.rank, self.NvS, stride, offset)
        launch_inter_rank_sync(self.ctx)

    def check_status(self):
        """Explicit local diagnostic only; synchronizes, never call in timing/capture."""
        code = int(self.status.item())
        if code:
            reasons = [text for bit, text in self.STATUS.items() if code & bit]
            raise RuntimeError(f"sampled materializer rejected candidate ({code}): {', '.join(reasons)}")

