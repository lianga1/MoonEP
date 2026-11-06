"""Source-local eight-rank bundle proposals and stable prefix quotas.

The native adapter supplies current all-gathered counts each round.
"""
from pathlib import Path
import hashlib
import operator
import torch
from torch.utils.cpp_extension import load

RANKS = 8
# Compatibility for old K=8 callers. New collectives must use instance.buckets.
BUCKETS = RANKS * RANKS * 2 * 9
_EXTENSIONS = {}

def extension(top_k=8):
    """One compiled specialization per K; expert count stays a runtime stride."""
    top_k = operator.index(top_k)
    if not 1 <= top_k <= 32:
        raise ValueError('top_k must be in 1..32')
    if top_k not in _EXTENSIONS:
        p = Path(__file__).parent / 'csrc' / 'sampled_exchange.cu'
        sha = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        _EXTENSIONS[top_k] = load(
            name=f'moonep_sampled_exchange_k{top_k}_{sha}', sources=[str(p)],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '-lineinfo', f'-DMOONEP_EXCHANGE_TOP_K={top_k}'],
            verbose=False,
        )
    return _EXTENSIONS[top_k]

class SampledExchange:
    """Eight source ranks, runtime E and compile-specialized K in 1..32.

    Each token proposes sizes 1, 2, 4, ... strictly below K, plus its entire
    old-owner group if that size was not already considered. Size preference
    rotates through these classes. K=8 retains the original four-class rule.
    Mask tensors are [S,8] int32 storage interpreted as unsigned 32-bit sets;
    their high bit is valid for expert-choice index 31. Owners remain [S,K].
    Routes must contain expert IDs in [0,E); seeds must contain ranks in [0,8).
    sm_limit=0 retains the original grid dimensions. A positive limit bounds
    every launch to that many CTAs while preserving virtual tile IDs/order.
    fuse=True enables the measured selective policy: small-table fold/quota
    remains fused, while proposal and stable ordinals stay separate with the
    default 256-token histogram. It requires a positive limit. The private
    _fuse_proposals override and stage 4 retain the old fused proposal for
    diagnostics; they do not change candidate or stable-order semantics.
    This class launches on the current stream;
    its limit does not constrain external collectives supplied by the caller.
    """
    def __init__(self,routes,support,*,source,rounds=4,sm_limit=0,fuse=False,_tile_tokens=256,_fuse_proposals=False):
        if (routes.ndim != 2 or routes.shape[0] <= 0 or not 1 <= routes.shape[1] <= 32
                or routes.dtype != torch.int32 or not routes.is_cuda or not routes.is_contiguous()):
            raise ValueError('routes require contiguous CUDA int32 [S,K], S>0 and 1<=K<=32')
        if (support.ndim != 2 or support.shape[0] != RANKS
                or not 8 <= support.shape[1] <= 4096 or support.shape[1] % RANKS
                or support.dtype != torch.bool or not support.is_contiguous() or support.device != routes.device):
            raise ValueError('support requires contiguous CUDA bool [8,E], 8<=E<=4096 and E divisible by 8')
        source, rounds = operator.index(source), operator.index(rounds)
        if source not in range(RANKS) or rounds < 1:
            raise ValueError('source 0..7 and positive rounds required')
        if routes.shape[0] > (2**31 - 1) // RANKS:
            raise ValueError('8*S must fit the int32 global count and token index contract')
        if isinstance(sm_limit, bool):
            raise ValueError('sm_limit must be a nonnegative integer, not bool')
        sm_limit = operator.index(sm_limit)
        if not 0 <= sm_limit <= torch.cuda.get_device_properties(routes.device).multi_processor_count:
            raise ValueError('sm_limit must be zero or no greater than this device SM count')
        if not isinstance(fuse, bool) or (fuse and sm_limit == 0):
            raise ValueError('fuse must be bool and requires a positive sm_limit')
        if not isinstance(_fuse_proposals, bool) or (_fuse_proposals and sm_limit == 0):
            raise ValueError('private proposal fusion must be bool and requires a positive sm_limit')
        self._fuse_proposals=_fuse_proposals
        if isinstance(_tile_tokens, bool):
            raise ValueError('private histogram tile must be 64, 128 or 256')
        _tile_tokens = operator.index(_tile_tokens)
        if _tile_tokens not in (64,128,256) or (_tile_tokens != 256 and sm_limit == 0):
            raise ValueError('private histogram tiles 64/128 require a positive sm_limit')
        self._tile_tokens=_tile_tokens
        self.sm_limit,self.fuse=sm_limit,fuse
        self.routes,self.support,self.source,self.rounds=routes,support,source,rounds
        self.S,self.K=routes.shape;self.E=support.shape[1]
        self.buckets=RANKS*RANKS*2*(self.K+1)
        self.blocks=(self.S+_tile_tokens-1)//_tile_tokens;self.ext=extension(self.K)
        def ints(*shape):return torch.empty(shape,device=routes.device,dtype=torch.int32)
        self.owner=torch.empty_like(routes);self.old_masks=ints(self.S,RANKS);self.exec_masks=ints(self.S,RANKS)
        self.resident=ints(self.E);self.keys=ints(self.S);self.moves=ints(self.S);self.ordinals=ints(self.S)
        self.block_counts=ints(self.buckets,self.blocks);self.block_prefix=ints(self.buckets,self.blocks)
        self.counts=ints(self.buckets);self.total=ints(self.buckets);self.source_prefix=ints(self.buckets);self.accept=ints(self.buckets)
        self.stats=torch.empty((rounds,4),device=routes.device,dtype=torch.int64)
    def prepare_support(self,support=None):
        support = self.support if support is None else support
        if (tuple(support.shape) != (RANKS,self.E) or support.dtype != torch.bool
                or support.device != self.routes.device or not support.is_contiguous()):
            raise ValueError('replacement support must match the original shape, dtype and device')
        self.ext.prepare(self.routes,support,self.resident,self.exec_masks,self.sm_limit)
    def reset(self,seed):
        if (seed.shape != self.owner.shape or seed.dtype != torch.int32
                or seed.device != self.owner.device or not seed.is_contiguous()):
            raise ValueError('seed must match the contiguous int32 [S,K] owner shape and device')
        self.ext.reset(seed,self.owner,self.old_masks,self.sm_limit)
    def _step(self,step):
        step = operator.index(step)
        if not 0 <= step < self.rounds:
            raise ValueError('step must be in [0,rounds)')
        return step
    def propose(self,step,*,_stage=0):
        """Generate proposals; private stage splits are for microbenchmarks only.

        0 runs the normal pipeline; 1 proposal, 2 ordinal, 3 prefix, 4 fused
        proposal+ordinal (explicitly overrides selective fusion). Stage 4
        requires a positive SM limit. Run stage 0 first to initialize inputs. Only
        stages 0/3 produce current counts. None of these stages changes owners.
        """
        if isinstance(_stage, bool):
            raise ValueError('private proposal stage must be in 0..4')
        _stage=operator.index(_stage)
        if _stage not in range(5) or (_stage==4 and self.sm_limit==0):
            raise ValueError('private proposal stage must be in 0..4; stage 4 requires a positive sm_limit')
        proposal_fuse=self._fuse_proposals or _stage==4
        self.ext.propose(self.old_masks,self.exec_masks,self.keys,self.moves,self.ordinals,self.block_counts,self.block_prefix,self.counts,self.S,self.source*self.S,self._step(step),self.sm_limit,proposal_fuse,self._tile_tokens,_stage)
        return self.counts
    def quota(self,step,gathered):
        if (gathered.numel() != RANKS*self.buckets or gathered.dtype != torch.int32
                or gathered.device != self.routes.device or not gathered.is_contiguous()):
            raise ValueError('gathered must contain 8*buckets contiguous CUDA int32 counts')
        self.ext.quota(gathered,self.total,self.source_prefix,self.accept,self.stats,self.source,self._step(step),self.sm_limit,self.fuse)
    def apply(self):
        self.ext.apply(self.keys,self.moves,self.ordinals,self.block_prefix,self.source_prefix,self.accept,self.old_masks,self.owner,self.sm_limit,self._tile_tokens)
        return self.owner
    def commit(self,step,gathered):self.quota(step,gathered);return self.apply()
