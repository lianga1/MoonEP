"""Split expert readiness from token readiness using real DeepEP v2.5 APIs.

No communication is emulated here. Importing/constructing this adapter requires a
working DeepEP extension; absence is an explicit unsupported result. The default
mode needs no DeepEP source change, but weight and token communication serialize
on DeepEP's global communication stream. Optional independent mode needs the
included lb-current-stream.patch and uses a second EPBuffer for isolated signals.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Sequence

import torch
import torch.distributed as dist

UPSTREAM_COMMIT = "f6aa38fe06cd4a0489e7730365d4e0114970a2f3"


def capabilities() -> dict:
    """Report actual import/API availability; never substitute a different backend."""
    result = {"backend": "DeepEP", "reference_commit": UPSTREAM_COMMIT}
    try:
        import deep_ep
        cls = getattr(deep_ep, "EPBuffer", None)
        result.update(version=getattr(deep_ep, "__version__", "unknown"),
                      path=deep_ep.__file__, epbuffer=cls is not None,
                      legacy_elastic=hasattr(deep_ep, "ElasticBuffer"))
        result["native_weight_prefetch"] = bool(cls and hasattr(cls, "lb_prefetch_weights"))
        result["independent_weight_stream_patch"] = bool(
            result["native_weight_prefetch"] and
            "use_current_stream" in inspect.signature(cls.lb_prefetch_weights).parameters)
    except Exception as exc:
        result.update(available=False, error=f"{type(exc).__name__}: {exc}")
        return result
    result["available"] = result["native_weight_prefetch"]
    return result


def physical_expert_ids(logical: torch.Tensor, owner: torch.Tensor,
                        redundancy: torch.Tensor, num_experts: int,
                        *, output_dtype=torch.int64) -> torch.Tensor:
    """Convert logical expert + execution rank to DeepEP's rank-major slot ID.

    Rank r owns slots [0,E/R) and replica slots [E/R,E/R+B). Expert r0*E/R+e
    placed on r1 in replica slot c becomes r1*(E/R+B)+E/R+c. Original logical
    IDs are retained outside the DeepEP handle. This pure GPU implementation is
    an integration reference; the final planner can write the same IDs directly.
    """
    ranks, extras = redundancy.shape
    if num_experts % ranks:
        raise ValueError("logical experts must be evenly sharded")
    if logical.shape != owner.shape or logical.ndim != 2:
        raise ValueError("logical and owner must have matching [tokens,topk] shapes")
    local = num_experts // ranks
    ids = torch.arange(num_experts, device=logical.device, dtype=torch.int64)
    rank_ids = torch.arange(ranks, device=logical.device, dtype=torch.int64)[:, None]
    table = torch.where(ids[None, :] // local == rank_ids,
                        ids[None, :] % local,
                        torch.full((ranks, num_experts), -1, device=logical.device,
                                   dtype=torch.int64))
    # Padding column absorbs unused -1 replica slots without overwriting expert 0.
    table = torch.cat((table, torch.full((ranks, 1), -1, device=logical.device,
                                       dtype=torch.int64)), dim=1)
    valid = redundancy >= 0
    scatter_ids = torch.where(valid, redundancy.to(torch.int64), num_experts)
    slots = (local + torch.arange(extras, device=logical.device,
                                  dtype=torch.int64))[None, :].expand(ranks, -1)
    table.scatter_(1, scatter_ids, torch.where(valid, slots, -1))
    selected = table[owner.to(torch.int64), logical.to(torch.int64)]
    return torch.where(selected >= 0, owner.to(torch.int64) * (local + extras) + selected,
                       -1).to(output_dtype).contiguous()


def validate_plan(logical, owner, redundancy, num_experts):
    """Synchronous correctness check, deliberately outside timed hot paths."""
    ranks, extras = redundancy.shape
    local = num_experts // ranks
    mapping = redundancy.detach().cpu()
    for rank in range(ranks):
        assigned = mapping[rank][mapping[rank] >= 0].tolist()
        if len(set(assigned)) != len(assigned):
            raise ValueError("duplicate expert in one rank's replica slots")
        if any(e >= num_experts or e // local == rank for e in assigned):
            raise ValueError("replica must refer to a remote logical expert")
    if not bool(((logical >= 0) & (logical < num_experts)).all()):
        raise ValueError("logical expert out of range")
    if not bool(((owner >= 0) & (owner < ranks)).all()):
        raise ValueError("execution rank out of range")
    physical = physical_expert_ids(logical, owner, redundancy, num_experts)
    if not bool((physical >= 0).all()):
        raise ValueError("execution plan requests an expert absent from target rank")
    return physical


@dataclass
class PendingWeights:
    event: Any
    # Keep mapping and sources alive until its event has been consumed.
    mapping: torch.Tensor
    sources: tuple
    destinations: tuple

    def wait(self):
        self.event.wait()


@dataclass
class PendingTokens:
    event: Any
    physical_ids: torch.Tensor
    logical_ids: torch.Tensor
    weights: torch.Tensor

    def wait(self):
        return self.event.wait()


class DeepEPPlannedAdapter:
    """A single-node group with externally supplied placement and token plans.

    Allocation and compilation belong outside a performance sample. All ranks
    invoke collectives in the same order, even if their replica row is all -1.
    One in-flight layer per buffer is supported. The caller must finish expert
    reads/gradient reduction before reusing weight storage for another version.
    """
    def __init__(self, group, *, per_source: int, hidden: int, topk: int,
                 num_experts: int, extra_slots: int,
                 weight_shapes: Sequence[tuple[int, ...]],
                 dtype=torch.bfloat16, num_sms: int = 16,
                 weight_num_sms: int = 8, expert_alignment: int = 128,
                 independent_weight_stream: bool = False):
        import deep_ep
        self.deep_ep = deep_ep
        cap = capabilities()
        if not cap["available"]:
            raise RuntimeError(f"DeepEP v2.5 native LB APIs unavailable: {cap}")
        if independent_weight_stream and not cap["independent_weight_stream_patch"]:
            raise RuntimeError("independent mode requires lb-current-stream.patch")
        self.group = group
        self.ranks, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if self.ranks != 2 or num_experts % self.ranks:
            raise ValueError("this experiment requires two equal expert shards")
        self.local_experts = num_experts // self.ranks
        self.num_experts, self.extra_slots = num_experts, extra_slots
        self.per_source, self.topk = per_source, topk
        self.num_sms, self.weight_num_sms = num_sms, weight_num_sms
        self.expert_alignment = expert_alignment
        self.independent_weight_stream = independent_weight_stream
        self.weight_stream = torch.cuda.Stream() if independent_weight_stream else None
        plan = deep_ep.BufferAllocator()
        self.replica_weights = tuple(plan.allocate((extra_slots, *shape), dtype)
                                     for shape in weight_shapes)
        common = dict(group=group, explicitly_destroy=True,
                      prefer_overlap_with_compute=True)
        if independent_weight_stream:
            # Separate context => separate workspace, entry/exit barriers and slots.
            self.weight_buffer = deep_ep.EPBuffer(
                **common, num_bytes=deep_ep.get_num_allocation_alignment(),
                lb_allocation_plan_or_num_bytes=plan)
            self.token_buffer = deep_ep.EPBuffer(
                **common, num_max_tokens_per_rank=per_source, hidden=hidden,
                num_topk=topk)
        else:
            self.token_buffer = self.weight_buffer = deep_ep.EPBuffer(
                **common, num_max_tokens_per_rank=per_source, hidden=hidden,
                num_topk=topk, lb_allocation_plan_or_num_bytes=plan)
        if self.weight_buffer.num_rdma_ranks != 1:
            raise ValueError("native replica prefetch requires one NVLink domain")
        self.expert_weights = tuple(torch.empty((self.local_experts, *shape),
                                                device="cuda", dtype=dtype)
                                    for shape in weight_shapes)

    def start_expert_prefetch(self, redundancy: torch.Tensor) -> PendingWeights:
        if redundancy.shape != (self.ranks, self.extra_slots):
            raise ValueError("redundancy shape differs from allocated slots")
        if redundancy.dtype != torch.int32 or not redundancy.is_contiguous():
            raise ValueError("redundancy must be contiguous CUDA int32")
        if self.independent_weight_stream:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
            self.weight_stream.wait_event(ready)
            with torch.cuda.stream(self.weight_stream):
                event = self.weight_buffer.lb_prefetch_weights(
                    self.replica_weights, self.expert_weights, redundancy,
                    num_sms=self.weight_num_sms, use_current_stream=True)
        else:
            event = self.weight_buffer.lb_prefetch_weights(
                self.replica_weights, self.expert_weights, redundancy,
                num_sms=self.weight_num_sms)
        return PendingWeights(event, redundancy, self.expert_weights, self.replica_weights)

    def start_token_dispatch(self, x, logical, owner, redundancy, topk_weights,
                             *, do_cpu_sync=False) -> PendingTokens:
        physical = physical_expert_ids(logical, owner, redundancy, self.num_experts,
                                        output_dtype=self.deep_ep.topk_idx_t)
        # Capture AFTER physical IDs are produced, but do not wait for weights.
        token_ready = self.token_buffer.capture()
        event = self.token_buffer.dispatch(
            x, topk_idx=physical, topk_weights=topk_weights,
            num_experts=self.ranks * (self.local_experts + self.extra_slots),
            num_max_tokens_per_rank=self.per_source,
            num_sms=self.num_sms, expert_alignment=self.expert_alignment,
            previous_event=token_ready, async_with_compute_stream=True,
            allocate_on_comm_stream=True, do_cpu_sync=do_cpu_sync,
            do_expand=True, do_zero_padding=True, defer_epilogue=True)
        return PendingTokens(event, physical, logical, topk_weights)

    @staticmethod
    def wait_for_expert_compute(weights: PendingWeights, tokens: PendingTokens):
        received = tokens.wait()
        weights.wait()
        return received

    def combine(self, weighted_expert_output, handle):
        # Router weights must already have been applied by the expert operation.
        return self.token_buffer.combine(
            weighted_expert_output, handle=handle, num_sms=self.num_sms,
            async_with_compute_stream=True, defer_epilogue=True)

    def destroy(self):
        torch.cuda.synchronize()
        if self.weight_buffer is not self.token_buffer:
            self.weight_buffer.destroy()
        self.token_buffer.destroy()
