"""Native C1-order reuse variant of the two-rank MoonEP output builder.

Same public interface as CudaMoonEPMaterializer. Requires a fresh native seed
for these routes before run(): ORDER_OFF and the expert histogram prefix are
read in place from its context. No copy/sort of full task IDs is performed.
"""
from pathlib import Path
import hashlib
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load
from moonep_materialize import extension as base_extension

_EXT=None

def extension():
    global _EXT
    if _EXT is None:
        path=Path(__file__).with_suffix('.cu');digest=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        _EXT=load(name='epmoe_native_order_reuse_'+digest,sources=[str(path)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
    return _EXT


class CudaMoonEPMaterializerReuse:
    def __init__(self,buffer,local_routes):
        from moonep.planning import allocate_planning_outputs
        self.ctx=buffer._require_ctx();c=self.ctx
        self.R,self.E,self.B,self.K,self.rank=[int(c[k]) for k in ('R','E','B','K','rank')]
        if self.R!=2 or self.E+self.B>512:raise ValueError('requires two ranks and E+B<=512')
        if local_routes.dtype!=torch.int32 or not local_routes.is_cuda or not local_routes.is_contiguous() or local_routes.numel()!=int(c['N']):raise ValueError('local routes must be contiguous CUDA int32 [S,K]')
        self.routes=local_routes.reshape(-1);M=self.routes.numel();self.Q=self.E+self.B;dev=local_routes.device
        def empty(*shape):return torch.empty(shape,dtype=torch.int32,device=dev)
        self.plan,self.cu=allocate_planning_outputs(c)
        self.native_owner=empty(int(c['S']),self.K);self.lut=empty(self.R,self.E)
        self.expert_starts=empty(self.E);self.expert_lengths=empty(self.E)
        self.local_ones=empty(M);blocks=(M+255)//256
        self.summary=empty(blocks,2);self.scanned=empty(blocks,2);self.last_expert=empty(blocks)
        self.counts=empty(self.R*self.Q);self.gathered=empty(self.R*self.R*self.Q)
        self.offsets=empty(self.R*self.Q);self.prefix=empty(self.R*self.Q);self.totals=empty(self.R*self.Q)
        self.raw=empty(M);self.errors=torch.zeros(4,dtype=torch.int32,device=dev)
        order_start=self.rank*int(c['meta_chunk_padded'])+int(c['ORDER_OFF'])
        self.order=c['meta_buf'][order_start:order_start+M]
        self.base_ext=base_extension();self.ext=extension()
        n=self.ext.workspace(self.summary,self.scanned)
        self.workspace=torch.empty(n,dtype=torch.uint8,device=dev)

    def decode_native_owner(self,plan):
        self.base_ext.decode_owner(plan.dst,self.native_owner,int(self.ctx['NvS']))
        return self.native_owner

    def prepare_support(self,support):
        self.base_ext.prepare_support(support,self.lut,self.plan.experts_to_copy,self.errors,self.E,self.B,self.R)
        return self.plan.experts_to_copy,torch.cuda.current_stream().record_event()

    def run(self,local_owner):
        from moonep.inter_rank_sync import launch_inter_rank_sync
        c=self.ctx
        self.ext.prepare(self.order,self.routes,local_owner,self.lut,self.expert_starts,self.expert_lengths,
            self.local_ones,self.summary,self.scanned,self.last_expert,self.workspace,self.counts,c['meta_buf'],self.errors,
            self.E,self.B,self.rank,int(c['NvS']),int(c['meta_chunk_padded']),int(c['PLAN_OFF']),int(c['SRC_INFO_OFF']))
        dist.all_gather_into_tensor(self.gathered,self.counts,group=c['group'])
        self.ext.publish(self.order,self.routes,local_owner,self.expert_starts,self.local_ones,self.scanned,self.last_expert,
            self.lut,self.gathered,self.offsets,self.prefix,self.totals,self.cu,self.plan.zero_fill_ranges,
            self.plan.physical_topk,self.raw,self.plan.dst,c['meta_buf'],self.plan.experts_to_copy,self.plan.remote_stats,self.errors,
            self.E,self.B,self.rank,self.K,int(c['token_padding']),int(c['NvS']),int(c['meta_chunk_padded']),int(c['SRC_INFO_OFF']))
        launch_inter_rank_sync(c)
        return self.plan,self.cu,torch.cuda.current_stream().record_event()

    def check_errors(self):
        values=self.errors.cpu().tolist()
        if any(values):raise RuntimeError(dict(zip(('support_overflow','invalid_task','capacity_overflow','unbalanced_load'),values)))
        return values
