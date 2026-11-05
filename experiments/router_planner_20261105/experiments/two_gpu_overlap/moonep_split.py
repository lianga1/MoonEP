"""Minimal split-phase orchestration for the supplied MoonEP source.

This module does not modify MoonEP kernels. It separates three dependencies:
final expert IDs -> weight copy; final token addresses -> token dispatch;
both completions -> expert computation. Native MoonEP's fused planner cannot
publish an event at its internal A5 boundary, so ``plan_native`` exposes only
whole-planner completion. An external planner can call ``prefetch`` earlier.

All buffers are single-flight. The caller must join compute/backward consumers
before overwriting plans/weight slots, and call close() before Buffer.destroy().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DispatchResult:
    hidden: torch.Tensor
    route_weights: torch.Tensor | None
    cu_seqlens: torch.Tensor
    plan: Any
    done: torch.cuda.Event


class SplitPhaseMoonEP:
    """Use one independent weight stream and MoonEP's existing token stream.

    Constructors and JIT compilation must be outside timed iterations. With a
    token-payload Buffer, native outputs and dispatch outputs are preallocated.
    ``prefetch_sms`` can reduce the weight-copy grid to leave SM capacity for
    planning; it does not promise that concurrent copies improve latency.
    """

    def __init__(self, buffer, *, prefetch_sms: int = 8,
                 expert_stream: torch.cuda.Stream | None = None):
        from moonep.planning import allocate_planning_outputs

        if prefetch_sms < 1:
            raise ValueError("prefetch_sms must be positive")
        self.buffer = buffer
        self.ctx = buffer._require_ctx()
        self.prefetch_sms = int(prefetch_sms)
        self.expert_stream = expert_stream or torch.cuda.Stream(
            device=int(self.ctx['device']), priority=0)
        self.token_stream = buffer._comm_stream
        if self.token_stream is None:
            raise RuntimeError("MoonEP token communication stream is unavailable")
        self.plan, self.cu_seqlens = allocate_planning_outputs(self.ctx)
        self._weight_done: torch.cuda.Event | None = None
        self._token_done: torch.cuda.Event | None = None
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("SplitPhaseMoonEP is closed")
        self.buffer._require_ctx()

    @staticmethod
    def _record(tensors, stream):
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(stream)

    def plan_native(self, topk_experts: torch.Tensor,
                    tokens_per_expert: torch.Tensor, *, inter_rank_sync=True):
        """Run native planning into reusable outputs on the current stream.

        This method does not initialize the plan-owned dedup arrays. Use
        dispatch_fresh(), never Buffer.dispatch(plan=...) for this first use.
        """
        from moonep.inter_rank_sync import launch_inter_rank_sync
        from moonep.planning import launch_planning

        self._check_open()
        if topk_experts.numel() != int(self.ctx['N']):
            raise ValueError("this fixed-shape adapter requires capacity-sized topk")
        if inter_rank_sync:
            launch_inter_rank_sync(self.ctx)
        launch_planning(self.ctx, topk_experts.reshape(-1), tokens_per_expert,
                        self.cu_seqlens, self.plan)
        ready = torch.cuda.current_stream().record_event()
        return self.plan, self.cu_seqlens, ready

    def prefetch(self, experts_to_copy: torch.Tensor, *,
                 ready: torch.cuda.Event,
                 full_gate_weight: torch.Tensor,
                 full_up_weight: torch.Tensor,
                 full_down_weight: torch.Tensor,
                 full_gate_scale: torch.Tensor | None = None,
                 full_up_scale: torch.Tensor | None = None,
                 full_down_scale: torch.Tensor | None = None,
                 weights_ready: torch.cuda.Event | None = None):
        """Start copies as soon as the final expert plan is ready.

        experts_to_copy is int32 [R,B], global IDs and -1 for empty slots.
        Its columns are physical destination slots and must remain stable
        through dispatch, expert computation and backward use. Inputs and
        source weight versions must remain immutable until the returned event.
        """
        from moonep.prefetch import _ELEM_TYPES, launch_prefetch, retile_for_prefetch

        self._check_open()
        R, E, B = (int(self.ctx[k]) for k in ('R', 'E', 'B'))
        if (experts_to_copy.dtype != torch.int32 or
                tuple(experts_to_copy.shape) != (R, B) or
                not experts_to_copy.is_contiguous()):
            raise ValueError("experts_to_copy must be contiguous int32 [R,B]")
        if not experts_to_copy.is_cuda:
            raise ValueError("expert plan must already be on the GPU")
        # The supplied API's gate/up arguments both refer to the fused W1.
        if full_gate_weight.data_ptr() != full_up_weight.data_ptr():
            raise ValueError("gate/up must alias the same fused W1 tensor")
        weights = (full_gate_weight, full_down_weight)
        for tensor in weights:
            if (tensor.ndim != 3 or tensor.shape[0] != E + B or
                    tensor.dtype not in _ELEM_TYPES or not tensor.is_contiguous()):
                raise ValueError("weights require contiguous supported [E+B,H,H']")
        scales = (full_gate_scale, full_up_scale, full_down_scale)
        if any(t is not None for t in scales):
            if not all(t is not None for t in scales):
                raise ValueError("all three scale arguments must be present together")
            if full_gate_scale.data_ptr() != full_up_scale.data_ptr():
                raise ValueError("gate/up scales must alias the same fused W1 scales")
            scales_to_copy = (full_gate_scale, full_down_scale)
        else:
            scales_to_copy = ()
        # Covers locally produced weights and their allocation lifetime. A
        # remote weight update requires its own distributed ready protocol.
        producer_ready = torch.cuda.current_stream().record_event()
        self.expert_stream.wait_event(ready)
        self.expert_stream.wait_event(producer_ready)
        if weights_ready is not None:
            self.expert_stream.wait_event(weights_ready)
        self._record((experts_to_copy, *weights, *scales_to_copy), self.expert_stream)
        with torch.cuda.stream(self.expert_stream):
            local_ids = experts_to_copy[int(self.ctx['rank'])]
            for tensor in weights:
                launch_prefetch(tensor[:E], tensor[E:], local_ids,
                                num_sms=self.prefetch_sms)
            for tensor in scales_to_copy:
                tiled = retile_for_prefetch(tensor)
                launch_prefetch(tiled[:E], tiled[E:], local_ids,
                                num_sms=self.prefetch_sms)
            self._weight_done = self.expert_stream.record_event()
        return self._weight_done

    def dispatch_fresh(self, hidden: torch.Tensor,
                       route_weights: torch.Tensor | None, *, plan=None,
                       cu_seqlens=None, ready: torch.cuda.Event,
                       inter_rank_sync=True):
        """Dispatch a newly generated plan, explicitly building its dedup map.

        Token dispatch waits on final token metadata, not on expert weights.
        Returns zero-copy views with MoonEP's normal aliasing lifetime: consume
        before the next dispatch/combine mutates the Buffer.
        """
        from moonep.dispatch import launch_dispatch
        from moonep.dispatch_epilogue import launch_dispatch_epilogue
        from moonep.inter_rank_sync import launch_inter_rank_sync

        self._check_open()
        self.buffer._require_token_payload(self.ctx)
        plan = self.plan if plan is None else plan
        cu_seqlens = self.cu_seqlens if cu_seqlens is None else cu_seqlens
        producer_ready = torch.cuda.current_stream().record_event()
        self.token_stream.wait_event(ready)
        self.token_stream.wait_event(producer_ready)
        tensors = (hidden, route_weights, cu_seqlens,
                   *self.buffer._plan_runtime_tensors(plan))
        self._record(tensors, self.token_stream)
        with torch.cuda.stream(self.token_stream):
            if inter_rank_sync:
                launch_inter_rank_sync(self.ctx)
            launch_dispatch(self.ctx, hidden, route_weights, plan,
                            build_dedup_map=True, pdl_trigger=False)
            launch_dispatch_epilogue(self.ctx, plan, pdl_launch=False)
            self._token_done = self.token_stream.record_event()
        output_weights = (None if route_weights is None else
                          self.ctx['weights_buf_local'].view(torch.float32))
        return DispatchResult(self.ctx['hidden_buf_local'], output_weights,
                              cu_seqlens, plan, self._token_done)

    def join(self, *, weight_done=None, token_done=None, stream=None):
        """Enqueue both dependencies before expert GEMM; does not block CPU."""
        self._check_open()
        stream = torch.cuda.current_stream() if stream is None else stream
        events = (self._weight_done if weight_done is None else weight_done,
                  self._token_done if token_done is None else token_done)
        if any(event is None for event in events):
            raise RuntimeError("both weight and token completions are required")
        for event in events:
            stream.wait_event(event)
        return stream.record_event()

    def close(self):
        """Drain independent copies before the owning Buffer/VMM is freed."""
        if not self._closed:
            self.expert_stream.synchronize()
            self.token_stream.synchronize()
            self._closed = True
