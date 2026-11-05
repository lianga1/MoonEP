"""Actual peer-mapped FP32 replica-gradient return using native MoonEP kernels.

Default fused layout reduces W1[2I,H] and W2[H,I] exactly once. The public
Buffer.reduce_grad API loops over three arguments without fused-W1 deduplication;
its optional separate layout therefore allocates distinct gate/up/down tensors.
Never pass an aliased fused W1 as both gate and up gradients to that API.

Native grad_reduce accepts FP32 only (both accumulation and NVLink payload).
BF16 model weights do not imply BF16 gradient traffic. BF16 producer gradients
must explicitly be converted into these FP32 buffers, with that conversion cost
accounted separately. This module never silently converts a BF16 transport run.

Version order: all local backward writes -> publish barrier -> owner reduction
and consumed-slot clear -> done event -> optimizer update -> globally publish
new weight version -> new prefetch. Keep the SAME expert/slot mapping alive from
forward prefetch through backward and gradient reduction. No optimizer/GEMM is
implemented here. reset/check are audit helpers OUTSIDE timed intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class GradientMatrix:
    name: str
    index: int
    home: torch.Tensor                 # physical local [E/R,h,hp]
    local_slots: torch.Tensor          # physical local [B,h,hp]
    full: torch.Tensor                 # virtual [E+B,h,hp]
    reduce: torch.Tensor               # virtual [R,B,h,hp], aliases local_slots
    home_row: torch.Tensor = field(repr=False)
    home_col: torch.Tensor = field(repr=False)
    slot_row: torch.Tensor = field(repr=False)
    slot_col: torch.Tensor = field(repr=False)


class GradientPool:
    def __init__(self, buffer, matrices, *, layout):
        self.buffer = buffer
        self.ctx = buffer._require_ctx()
        self.E, self.B, self.R, self.rank = [int(self.ctx[k]) for k in ('E','B','R','rank')]
        self.matrices, self.layout = matrices, layout
        self.dtype = torch.float32
        self.epoch = None
        self._done = None
        self._closed = False

    @property
    def bytes_per_expert(self):
        return sum(m.full[0].numel()*4 for m in self.matrices)

    def precompile(self):
        """Compile without launches, then align ranks before any native barrier."""
        from moonep.grad_reduce import _get_compiled
        from moonep import inter_rank_sync
        c = self.ctx
        for m in self.matrices:
            h,hp = map(int,m.full.shape[1:])
            _get_compiled(self.E,h,hp,self.R,self.B,int(c['meta_chunk_padded']),
                          int(c['num_sms']),int(c['device']))
        inter_rank_sync._get_compiled(self.R,int(c['meta_chunk_padded']),
                                     inter_rank_sync._num_threads_for_ranks(self.R),int(c['device']))
        dist.barrier(group=c['group'])

    def reset(self, epoch=0):
        """GPU-only local writes; no host/device sync or distributed collective.

        Call OUTSIDE timing before EACH cold/training sample. The epoch is an
        audit fingerprint, not an optimizer operation or model weight version.
        """
        if self._closed: raise RuntimeError('GradientPool is closed')
        if type(epoch) is not int or epoch < 0: raise ValueError('epoch must be a nonnegative integer')
        if self._done is not None: torch.cuda.current_stream().wait_event(self._done)
        self.epoch = epoch
        lo = self.rank*(self.E//self.R)
        for m in self.matrices:
            ids = torch.arange(lo,lo+self.E//self.R,device=m.home.device,dtype=torch.float32)
            base = ids*17 + m.index*1009 + epoch*19
            m.home.copy_(base[:,None,None].expand_as(m.home))
            m.home.add_(m.home_row);m.home.add_(m.home_col)
            slots = torch.arange(self.B,device=m.home.device,dtype=torch.float32)
            base = (self.rank+1)*101 + slots*11 + m.index*1237 + epoch*23
            m.local_slots.copy_(base[:,None,None].expand_as(m.local_slots))
            m.local_slots.add_(m.slot_row);m.local_slots.add_(m.slot_col)

    def launch(self, plan_or_experts, *, ready=None, publish=True):
        """Enqueue on Buffer's comm stream; caller MUST join returned event.

        The default GPU publish barrier is inside the timed path. Native reduce
        only fences peers AFTER its reads, so it cannot replace this input-ready
        barrier. publish=False requires a separately established global ready
        protocol. No CPU readback, allocation, or correctness check occurs here.
        """
        if self._closed: raise RuntimeError('GradientPool is closed')
        from moonep.grad_reduce import launch_grad_reduce
        from moonep.inter_rank_sync import launch_inter_rank_sync
        plan = plan_or_experts if hasattr(plan_or_experts,'experts_to_copy') else None
        experts = plan.experts_to_copy if plan is not None else plan_or_experts
        if experts.dtype != torch.int32 or tuple(experts.shape)!=(self.R,self.B) or not experts.is_cuda or not experts.is_contiguous():
            raise ValueError('expert mapping requires contiguous CUDA int32 [R,B]')
        if self.layout == 'separate' and plan is None:
            raise ValueError('separate layout uses Buffer.reduce_grad and requires a MoonEPCommPlan')
        stream = self.buffer._comm_stream
        if stream is None: raise RuntimeError('MoonEP communication stream is unavailable')
        producer = torch.cuda.current_stream().record_event()
        stream.wait_event(producer)
        if ready is not None: stream.wait_event(ready)
        experts.record_stream(stream)
        for m in self.matrices:
            for tensor in (m.home,m.local_slots,m.full,m.reduce): tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            if publish: launch_inter_rank_sync(self.ctx)
            if self.layout == 'separate':
                kwargs = {}
                for m in self.matrices:
                    kwargs['full_'+m.name+'_grad'] = m.full
                    kwargs[m.name+'_reduce_buffer'] = m.reduce
                self.buffer.reduce_grad(plan=plan,async_finish=False,**kwargs)
            else:
                c = self.ctx
                for m in self.matrices:
                    launch_grad_reduce(m.full[:self.E],m.reduce,experts,rank=self.rank,
                        num_sms=int(c['num_sms']),meta_buf=c['meta_buf'],
                        meta_stride=int(c['meta_chunk_padded']),barrier_off=int(c['BARRIER_OFF']),
                        grid_sync_bar=c['grid_sync_bar'])
            self._done = stream.record_event()
        return self._done

    def check(self, experts_to_copy):
        """Full exact check, outside timing; includes inactive homes and slots."""
        if self.epoch is None: raise RuntimeError('reset() must establish the reference gradient values')
        if hasattr(experts_to_copy,'experts_to_copy'): experts_to_copy=experts_to_copy.experts_to_copy
        if self._done is not None: self._done.synchronize()
        torch.cuda.synchronize();dist.barrier(group=self.ctx['group']);torch.cuda.synchronize()
        mapping = experts_to_copy.cpu().tolist()
        if len(mapping)!=self.R or any(len(row)!=self.B for row in mapping): raise ValueError('bad mapping shape')
        if any(e < -1 or e >= self.E for row in mapping for e in row): raise ValueError('invalid expert ID')
        epn=self.E//self.R;lo=self.rank*epn
        checks=[];counts=dict(active_home=0,inactive_home=0,active_slots=0,inactive_slots=0)
        additions={e:[] for e in range(lo,lo+epn)}
        for source,row in enumerate(mapping):
            for b,e in enumerate(row):
                if e in additions: additions[e].append((source,b))
        for e,slots in additions.items(): counts['active_home' if slots else 'inactive_home']+=1
        for m in self.matrices:
            for e,slots in additions.items():
                expected=torch.empty(m.full.shape[1:],device=m.full.device,dtype=torch.float32)
                # Match reset's addition order, and then native's rb-ascending
                # FP32 accumulation order; this also works when additions round.
                expected.fill_(e*17+m.index*1009+self.epoch*19)
                expected.add_(m.home_row);expected.add_(m.home_col)
                for source,b in slots:
                    slot=torch.empty_like(expected)
                    slot.fill_((source+1)*101+b*11+m.index*1237+self.epoch*23)
                    slot.add_(m.slot_row);slot.add_(m.slot_col)
                    expected.add_(slot)
                checks.append(torch.all(m.full[e]==expected))
            for b,e in enumerate(mapping[self.rank]):
                if m.index==0: counts['active_slots' if e>=0 else 'inactive_slots']+=1
                expected=torch.empty_like(m.local_slots[b])
                if e>=0: expected.zero_()
                else:
                    expected.fill_((self.rank+1)*101+b*11+m.index*1237+self.epoch*23)
                    expected.add_(m.slot_row);expected.add_(m.slot_col)
                # Verify all three virtual/physical aliases of this rank's slot.
                checks.extend((torch.all(m.local_slots[b]==expected),torch.all(m.reduce[self.rank,b]==expected),
                               torch.all(m.full[self.E+b]==expected)))
        ok=torch.stack(checks).all().to(torch.int32)
        dist.all_reduce(ok,op=dist.ReduceOp.MIN,group=self.ctx['group'])
        if not bool(ok.item()): raise AssertionError('native FP32 gradient sum/slot-clear/inactive preservation failed on at least one rank')
        return dict(rank=self.rank,layout=self.layout,dtype='float32',epoch=self.epoch,
                    full_home_and_slot_check=True,inactive_preserved=True,active_slots_cleared=True,
                    bytes_per_expert=self.bytes_per_expert,**counts)

    def source_manifest(self):
        import moonep.grad_reduce as grad_reduce
        import moonep.api as api
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),Path(grad_reduce.__file__),Path(api.__file__))}

    def close(self):
        if not self._closed:
            if self._done is not None:self._done.synchronize()
            torch.cuda.synchronize()
            self.matrices.clear();self._done=None;self._closed=True


def allocate_gradients(buffer,H,I,*,layout='fused',dtype=torch.float32):
    """Allocate real home gradients and peer-visible replica accumulation slots.

    full[E:E+B] and reduce[rank] map the SAME physical allocation. Source homes
    and replica slots live only on their owning GPU, as with WeightPool; no full
    replicated gradient tensor is hidden in this allocation.
    """
    if dtype!=torch.float32:
        raise ValueError('native MoonEP grad_reduce supports FP32 only; BF16 transport requires a different kernel')
    if layout not in ('fused','separate'):raise ValueError("layout must be 'fused' or 'separate'")
    if H<=0 or I<=0 or H%128 or I%128:raise ValueError('native gradients require positive H/I multiples of 128')
    from moonep._C import nvl_dist_alloc,nvl_dist_map_segments,nvl_release_mem_handle,get_vmm_granularity
    from moonep.buffer import _use_fabric_for_group,_all_gather_shareables,_exchange_ipc_fds
    c=buffer._require_ctx();E,B,R,rank=[int(c[k]) for k in ('E','B','R','rank')]
    if E%R or B<1:raise ValueError('E must be divisible by R and B must be positive')
    epn=E//R;fabric=_use_fabric_for_group(c['group']);gran=get_vmm_granularity()
    shapes=(('W1',2*I,H),('W2',H,I)) if layout=='fused' else (('gate',I,H),('up',I,H),('down',H,I))
    matrices=[]
    for index,(name,h,hp) in enumerate(shapes):
        homebytes=epn*h*hp*4;slotbytes=B*h*hp*4
        if homebytes%gran or slotbytes%gran:raise ValueError('gradient home/slot allocations must satisfy VMM alignment')
        home,hs,ho=nvl_dist_alloc(shape=[epn,h,hp],dtype=dtype,use_fabric=fabric)
        local,ss,so=nvl_dist_alloc(shape=[B,h,hp],dtype=dtype,use_fabric=fabric)
        nvl_release_mem_handle(ho);nvl_release_mem_handle(so)
        fds=[];export_fds=[]
        try:
            if fabric:
                homes=_all_gather_shareables(hs,c['group']);slots=_all_gather_shareables(ss,c['group'])
                full_handles=torch.cat((homes,slots[rank:rank+1]),dim=0)
            else:
                hfd,sfd=int(hs.item()),int(ss.item())
                export_fds.extend((hfd,sfd))
                try:home_map=_exchange_ipc_fds(hfd,list(range(R)),rank,R,c['group'])
                finally:os.close(hfd);export_fds.remove(hfd)
                fds.extend(home_map.values())
                try:slot_map=_exchange_ipc_fds(sfd,list(range(R)),rank,R,c['group'])
                finally:os.close(sfd);export_fds.remove(sfd)
                fds.extend(slot_map.values())
                homes=torch.tensor([home_map[r] for r in range(R)],dtype=torch.int64)
                slots=torch.tensor([slot_map[r] for r in range(R)],dtype=torch.int64)
                full_handles=torch.cat((homes,slots[rank:rank+1]))
            full=nvl_dist_map_segments(shape=[E+B,h,hp],dtype=dtype,shareables=full_handles,
                segment_bytes=[homebytes]*R+[slotbytes],destination_offsets=[r*homebytes for r in range(R)]+[E*h*hp*4],use_fabric=fabric)
            reduce=nvl_dist_map_segments(shape=[R*B,h,hp],dtype=dtype,shareables=slots,
                segment_bytes=[slotbytes]*R,destination_offsets=[r*slotbytes for r in range(R)],use_fabric=fabric).view(R,B,h,hp)
        finally:
            for fd in fds:os.close(fd)
            for fd in export_fds:os.close(fd)
        row=torch.arange(h,device=home.device,dtype=torch.float32).view(h,1)
        col=torch.arange(hp,device=home.device,dtype=torch.float32).view(1,hp)
        matrices.append(GradientMatrix(name,index,home,local,full,reduce,
            (row%17)*0.25,(col%13)*0.015625,(row%19)*0.125,(col%11)*0.0078125))
    return GradientPool(buffer,matrices,layout=layout)
