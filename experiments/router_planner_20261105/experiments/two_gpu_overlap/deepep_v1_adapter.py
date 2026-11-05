"""DeepEP v1.2.1 intranode token path with external expert placement.

The official v1.2.1 base uses separately recorded local build fixes and an
intranode-notify launch-capacity patch for 184 local slots. A physical expert ID
encodes the planner's execution rank and local slot. Weight transport is separately identified:
NCCLReplicaPrefetch is a Torch/NCCL reference, NOT a DeepEP weight kernel.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from deepep_adapter import physical_expert_ids

UPSTREAM_COMMIT = "9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee"


def build_provenance():
    """Record the actual imported extension as well as local source/patch hashes."""
    import deep_ep
    import deep_ep_cpp
    root = Path(__file__).resolve().parent
    paths = [Path(deep_ep.__file__), Path(deep_ep_cpp.__file__),
             root / 'vendor/DeepEP_v1/setup.py',
             root / 'vendor/DeepEP_v1/csrc/kernels/intranode.cu',
             root / 'deepep-v1-disable-nvshmem.patch',
             root / 'deepep-v1-intranode-dlink.patch',
             root / 'deepep-v1-notify-256.patch']
    return dict(upstream_commit=UPSTREAM_COMMIT,
                local_variant='build fixes plus notify launch 128/256 threads',
                files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths if p.is_file()})


@dataclass
class V1TokenTicket:
    result: tuple
    physical: torch.Tensor
    logical: torch.Tensor
    topk_weights: torch.Tensor

    def wait(self):
        *values, event = self.result
        event.current_stream_wait()
        return tuple(values)


@dataclass
class WeightTicket:
    event: Any
    keepalive: tuple

    def wait(self):
        torch.cuda.current_stream().wait_event(self.event)


class NCCLReplicaPrefetch:
    """Reference weight path with an explicit, already host-visible plan.

    This constructor's CPU plan conversion is preparation, not free: if the
    online GPU planner changes the map, include its D2H sync + construction in
    full planner timing. Prefer the MoonEP one-sided GPU-indexed transport for a
    fully GPU-resident path. Both ranks construct/call in identical order.
    """
    def __init__(self, group, expert_weights, replica_weights, mapping_cpu):
        self.group = group
        self.rank, self.ranks = dist.get_rank(group), dist.get_world_size(group)
        if self.ranks != 2 or mapping_cpu.device.type != "cpu":
            raise ValueError("two-rank precompiled CPU mapping required")
        self.expert_weights, self.replica_weights = expert_weights, replica_weights
        self.mapping = mapping_cpu.clone()
        self.local_experts = expert_weights.shape[0]
        self.stream = torch.cuda.Stream()
        self.ops = []
        self.bytes_sent = self.bytes_received = 0
        bytes_per = expert_weights[0].numel() * expert_weights.element_size()
        for destination in range(self.ranks):
            for slot, expert in enumerate(self.mapping[destination].tolist()):
                if expert < 0:
                    continue
                source = expert // self.local_experts
                if source == destination:
                    raise ValueError("local experts must not occupy replica slots")
                if source == self.rank:
                    self.ops.append(dist.P2POp(dist.isend,
                        expert_weights[expert % self.local_experts], destination,
                        group=group))
                    self.bytes_sent += bytes_per
                elif destination == self.rank:
                    self.ops.append(dist.P2POp(dist.irecv, replica_weights[slot], source,
                                               group=group))
                    self.bytes_received += bytes_per

    def start(self):
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        self.stream.wait_event(ready)
        with torch.cuda.stream(self.stream):
            requests = dist.batch_isend_irecv(self.ops) if self.ops else []
            for request in requests:
                request.wait()
            done = torch.cuda.Event()
            done.record(self.stream)
        return WeightTicket(done, (self.expert_weights, self.replica_weights,
                                    self.mapping, requests))


class DeepEPV1PlannedAdapter:
    def __init__(self, group, *, per_source, hidden, topk=8, num_experts=352,
                 extra_slots=8, num_sms=16):
        # Conservative boundary for this BF16 adapter. H=128 failed combine;
        # upstream's TMA loop uses full-warp synchronization inside a lane-index
        # loop. A multiple of 256 makes hidden*sizeof(BF16)/sizeof(int4) divisible
        # by 32. This avoids the known failing shape; it is not a proven kernel
        # repair or validation of every other accepted size. Target tests use 7168.
        if hidden < 256 or hidden % 256:
            raise ValueError('DeepEP v1 BF16 adapter requires hidden divisible by 256; '
                             'H=128 is a retained combine failure; target validation uses H=7168')
        import deep_ep
        if not hasattr(deep_ep, "Buffer"):
            raise RuntimeError("DeepEP v1 Buffer API required")
        self.deep_ep = deep_ep
        self.group, self.ranks = group, dist.get_world_size(group)
        if self.ranks != 2:
            raise ValueError("this reference requires two physical ranks")
        self.per_source, self.num_experts = per_source, num_experts
        self.extra_slots = extra_slots
        self.physical_experts = num_experts + self.ranks * extra_slots
        deep_ep.Buffer.set_num_sms(num_sms)
        self.dispatch_config = deep_ep.Buffer.get_dispatch_config(self.ranks)
        self.combine_config = deep_ep.Buffer.get_combine_config(self.ranks)
        nbytes = max(self.dispatch_config.get_nvl_buffer_size_hint(hidden * 2, self.ranks),
                     self.combine_config.get_nvl_buffer_size_hint(hidden * 2, self.ranks))
        self.buffer = deep_ep.Buffer(group, num_nvl_bytes=nbytes, num_rdma_bytes=0,
                                     explicitly_destroy=True)

    def start_token_dispatch(self, x, logical, owner, redundancy, topk_weights,
                             *, no_cpu_sync=True):
        physical = physical_expert_ids(logical, owner, redundancy, self.num_experts)
        return self.start_physical_dispatch(x, physical, topk_weights,
                                            logical=logical, no_cpu_sync=no_cpu_sync)

    def start_physical_dispatch(self, x, physical, topk_weights, *, logical=None,
                                no_cpu_sync=True):
        """Skip adapter remapping when the planner already emits physical IDs."""
        ready = self.buffer.capture()
        per_rank, _, per_expert, in_rank, layout_ready = self.buffer.get_dispatch_layout(
            physical, self.physical_experts, previous_event=ready,
            async_finish=True, allocate_on_comm_stream=True)
        result = self.buffer.dispatch(
            x, topk_idx=physical, topk_weights=topk_weights,
            num_tokens_per_rank=per_rank, num_tokens_per_expert=per_expert,
            is_token_in_rank=in_rank, previous_event=layout_ready,
            async_finish=True, allocate_on_comm_stream=True,
            num_worst_tokens=self.ranks * self.per_source if no_cpu_sync else 0,
            config=self.dispatch_config)
        return V1TokenTicket(result, physical, logical, topk_weights)

    def combine(self, weighted_rank_output, handle):
        # Each deduplicated receive row already sums this rank's expert outputs.
        out, weights, event = self.buffer.combine(
            weighted_rank_output, handle=handle, previous_event=self.buffer.capture(),
            async_finish=True, allocate_on_comm_stream=True, config=self.combine_config)
        return out, weights, event

    def destroy(self):
        torch.cuda.synchronize()
        self.buffer.destroy()
