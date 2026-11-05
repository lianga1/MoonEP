"""Shared native MoonEP compile and real peer-mapped weight helpers."""
from dataclasses import dataclass, field
import os
import torch
import torch.distributed as dist


def create_buffer(S,H,K=8,E=352,B=8,num_sms=32):
    from moonep import Buffer
    return Buffer(S,H,K,E,dist.get_world_size(),B=B,num_sms=num_sms,
                  replica_target=0,replica_max=min(8,B-1),enable_pdl=False,
                  explicitly_destroy=True)


def compile_native(buffer,prefetch_sms=8,weight_shapes=None):
    """Compile every native collective before any rank launches one.

    This prevents one rank entering a native 100-second spin barrier while
    its peer is still in a long first-time JIT compilation. All work is untimed.
    weight_shapes is optional [(E,H,Hp,B,dtype), ...].
    """
    from moonep import planning,inter_rank_sync,dispatch,dispatch_epilogue,combine,combine_prologue,prefetch
    c=buffer._require_ctx()
    R,E,B,S,K,H,NvS,NvSp,ms,ns,dev=[int(c[k]) for k in ('R','E','B','S','K','H','NvS','NvS_padded','meta_chunk_padded','num_sms','device')]
    planning._get_compiled(R,E,B,S,K,int(c['NvS_capacity']),NvS,int(c['num_vblocks']),ms,
        int(c['TPE_OFF']),int(c['PLAN_OFF']),int(c['BARRIER_OFF']),int(c['TOPK0_OFF']),
        int(c['ORDER_OFF']),int(c['ORDER0_OFF']),int(c['replica_max']),int(c['token_padding']),ns)
    inter_rank_sync._get_compiled(R,ms,inter_rank_sync._num_threads_for_ranks(R),dev)
    if not c['planner_only']:
        for with_weights in (False,True):
            for fresh in (False,True):
                dispatch._get_compiled(H,R,S,K,E+B,NvS,NvSp,ms,int(c['SRC_INFO_OFF']),ns,with_weights,fresh,dev,False)
            combine._get_compiled(H,R,S,K,NvS,NvSp,ms,ns,with_weights,dev,False)
        dispatch_epilogue._get_compiled(H,NvS,int(c['num_sms_dedup']),dev,False)
        combine_prologue._get_compiled(H,R,NvS,int(c['num_sms_dedup']),dev,False)
    for e,h,hp,b,dtype in weight_shapes or ():
        prefetch._get_compiled(e,h,hp,b,prefetch_sms,dev,dtype)
    dist.barrier(group=c['group'])


@dataclass
class WeightPool:
    W1:torch.Tensor
    W2:torch.Tensor
    E:int
    B:int
    rank:int
    R:int
    keepalives:list=field(repr=False)

    def kwargs(self):
        return dict(full_gate_weight=self.W1,full_up_weight=self.W1,full_down_weight=self.W2)

    @property
    def bytes_per_expert(self):
        return sum(t[0].numel()*t.element_size() for t in (self.W1,self.W2))

    def fill_sources(self):
        epn=self.E//self.R;lo=self.rank*epn;hi=lo+epn
        expert_ids=torch.arange(lo,hi,device=self.W1.device,dtype=torch.int32)
        values=(expert_ids%31+1).to(torch.bfloat16).view(epn,1,1)
        quotients=(expert_ids//31+1).to(torch.bfloat16).view(epn,1,1)
        self.W1[lo:hi].copy_(values.expand_as(self.W1[lo:hi]))
        self.W2[lo:hi].copy_(quotients.expand_as(self.W2[lo:hi]))
        self.W1[self.E:].fill_(-1);self.W2[self.E:].fill_(-1)
        torch.cuda.synchronize();dist.barrier()

    def close(self):
        torch.cuda.synchronize()
        self.W1=None;self.W2=None;self.keepalives.clear()


def allocate_weights(buffer,H,I):
    """Map R unequal home chunks plus this rank's B-slot destination chunk.

    Sources are real local expert shards, not replicated full weight tensors.
    Mapping [E+B,...] is virtual; remote source rows traverse the peer link.
    """
    from moonep._C import nvl_dist_alloc,nvl_dist_map_segments,nvl_release_mem_handle,get_vmm_granularity,FABRIC_HANDLE_BYTES
    from moonep.buffer import _use_fabric_for_group,_all_gather_shareables,_exchange_ipc_fds
    c=buffer._require_ctx();E,B,R,rank=[int(c[k]) for k in ('E','B','R','rank')];epn=E//R
    use_fabric=_use_fabric_for_group(c['group']);keep=[]
    def build(h,hp):
        rowbytes=h*hp*2;sourcebytes=epn*rowbytes;copybytes=B*rowbytes;gran=get_vmm_granularity()
        if sourcebytes%gran or copybytes%gran:
            raise ValueError(f'weight segments not VMM aligned: home={sourcebytes},copies={copybytes},gran={gran}; H=128 smoke can use I=1024')
        home,hs,ho=nvl_dist_alloc(shape=[epn,h,hp],dtype=torch.bfloat16,use_fabric=use_fabric)
        copies,cs,co=nvl_dist_alloc(shape=[B,h,hp],dtype=torch.bfloat16,use_fabric=use_fabric)
        keep.extend((home,copies));nvl_release_mem_handle(ho);nvl_release_mem_handle(co)
        sizes=[sourcebytes]*R+[copybytes];offsets=[r*sourcebytes for r in range(R)]+[E*rowbytes]
        if use_fabric:
            all_h=_all_gather_shareables(hs,c['group'])
            shares=torch.cat((all_h,cs.view(1,FABRIC_HANDLE_BYTES)),dim=0)
            full=nvl_dist_map_segments(shape=[E+B,h,hp],dtype=torch.bfloat16,shareables=shares,
                    segment_bytes=sizes,destination_offsets=offsets,use_fabric=True)
        else:
            hfd,cfd=int(hs.item()),int(cs.item())
            fds=_exchange_ipc_fds(hfd,list(range(R)),rank,R,c['group'])
            os.close(hfd)
            received=[fds[r] for r in range(R)]
            try:
                full=nvl_dist_map_segments(shape=[E+B,h,hp],dtype=torch.bfloat16,
                    shareables=torch.tensor(received+[cfd],dtype=torch.int64),segment_bytes=sizes,
                    destination_offsets=offsets,use_fabric=False)
            finally:
                for fd in received:os.close(fd)
                os.close(cfd)
        return full
    pool=WeightPool(build(2*I,H),build(H,I),E,B,rank,R,keep)
    return pool


def check_weight_slots(pool,experts_to_copy):
    """Untimed full slot check against the expert-ID constants used by fill."""
    ids=experts_to_copy[pool.rank].cpu().tolist();checked=0
    for b,e in enumerate(ids):
        if e<0:continue
        assert torch.all(pool.W1[pool.E+b]==e%31+1).item(),('W1',pool.rank,b,e)
        assert torch.all(pool.W2[pool.E+b]==e//31+1).item(),('W2',pool.rank,b,e)
        checked+=1
    return dict(checked_active_slots=checked,bytes_per_expert=pool.bytes_per_expert)
