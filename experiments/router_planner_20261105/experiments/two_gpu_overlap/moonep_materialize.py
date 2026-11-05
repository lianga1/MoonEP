"""Source-sharded CUDA owner/support -> native MoonEP dispatch plan.

Only R*(E+B) int32 counts per source are gathered. Stable local sort plus
source-rank prefixes reproduces global source/token/top-k occurrence order.
Experts use native E+B segment rows, while physical_topk uses compact E/R+B
indices. Writes source provenance directly into MoonEP's peer-mapped meta_buf.
"""
from pathlib import Path
import hashlib
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        path = Path(__file__).with_suffix('.cu')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        _EXT = load(name='epmoe_native_materialize_' + digest, sources=[str(path)],
                    extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
                    verbose=False)
    return _EXT


class CudaMoonEPMaterializer:
    def __init__(self, buffer, local_routes):
        from moonep.planning import allocate_planning_outputs
        self.ctx = buffer._require_ctx()
        self.routes = local_routes.reshape(-1)
        self.R,self.E,self.B,self.K,self.rank = [int(self.ctx[k]) for k in ('R','E','B','K','rank')]
        if self.R != 2:
            raise ValueError('this first audited integration targets two physical ranks')
        if (local_routes.dtype != torch.int32 or not local_routes.is_cuda or
                not local_routes.is_contiguous() or self.routes.numel()!=int(self.ctx['N'])):
            raise ValueError('routes must be local contiguous int32 [S,K] at context capacity')
        if self.E+self.B>512:
            raise ValueError('native segment layout currently requires E+B<=512')
        self.Q=self.E+self.B
        self.bits=(self.R*self.Q).bit_length()  # includes invalid-key sentinel
        dev=local_routes.device
        def empty(*shape):return torch.empty(shape,dtype=torch.int32,device=dev)
        self.plan,self.cu=allocate_planning_outputs(self.ctx)
        self.lut=empty(self.R,self.E)
        self.ids=torch.arange(self.routes.numel(),device=dev,dtype=torch.int32)
        self.keys=empty(self.routes.numel());self.sorted_keys=empty(self.routes.numel());self.sorted_ids=empty(self.routes.numel())
        self.starts=empty(self.R*self.Q);self.counts=empty(self.R*self.Q)
        self.gathered=empty(self.R*self.R*self.Q)
        self.offsets=empty(self.R*self.Q);self.prefix=empty(self.R*self.Q);self.totals=empty(self.R*self.Q)
        self.native_owner=empty(int(self.ctx['S']),self.K)
        self.raw=empty(self.routes.numel());self.errors=torch.zeros(4,dtype=torch.int32,device=dev)
        self.ext=extension()
        size=self.ext.workspace(self.keys,self.sorted_keys,self.ids,self.sorted_ids,self.bits)
        self.workspace=torch.empty(size,dtype=torch.uint8,device=dev)

    def decode_native_owner(self,plan):
        """Decode signed native dst directly into reusable local int32 [S,K]."""
        self.ext.decode_owner(plan.dst,self.native_owner,int(self.ctx['NvS']))
        return self.native_owner

    def prepare_support(self,support):
        """Finalize stable extra-slot order; call as soon as support is fixed.

        Supports are replicated identical bool or int32 [R,E]. Retain this returned
        expert table unchanged until prefetch, execution and backward finish.
        """
        self.ext.prepare_support(support,self.lut,self.plan.experts_to_copy,self.errors,
                                 self.E,self.B,self.R)
        return self.plan.experts_to_copy,torch.cuda.current_stream().record_event()

    def run(self,local_owner):
        """Return a fresh plan; its dedup builder must run on first dispatch."""
        from moonep.inter_rank_sync import launch_inter_rank_sync
        c=self.ctx
        self.ext.local_prepare(self.routes,local_owner,self.lut,self.ids,self.keys,
            self.plan.physical_topk,self.sorted_keys,self.sorted_ids,self.workspace,
            self.starts,self.counts,c['meta_buf'],self.errors,self.E,self.B,self.R,
            self.rank,int(c['NvS']),int(c['meta_chunk_padded']),int(c['SRC_INFO_OFF']),self.bits)
        # Previous local SRC_INFO clear precedes this collective. Every rank
        # finishes clearing before any following peer provenance publication.
        dist.all_gather_into_tensor(self.gathered,self.counts,group=c['group'])
        self.ext.publish(local_owner,self.sorted_keys,self.sorted_ids,self.starts,
            self.gathered,self.offsets,self.prefix,self.totals,self.cu,
            self.plan.zero_fill_ranges,self.raw,self.plan.dst,c['meta_buf'],
            self.plan.experts_to_copy,self.plan.remote_stats,self.errors,
            self.E,self.B,self.R,self.rank,self.K,int(c['token_padding']),
            int(c['NvS']),int(c['meta_chunk_padded']),int(c['SRC_INFO_OFF']))
        launch_inter_rank_sync(c)
        return self.plan,self.cu,torch.cuda.current_stream().record_event()

    def check_errors(self):
        values=self.errors.cpu().tolist()
        if any(values):
            raise RuntimeError(dict(zip(('support_overflow','invalid_task','capacity_overflow','unbalanced_load'),values)))
        return values
