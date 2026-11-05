"""Native eight-rank sampled-residency / balanced bundle-exchange adapter.

Each process owns only its original source tokens. Live sample and count
collectives execute on the current stream; no trace inputs or CPU routing
choices appear in the dispatch path. Scratch tensors belong to this Buffer,
whereas all final results are copied into the caller's MoonEPCommPlan.
"""
import torch
import torch.distributed as dist

from .sampled_exchange import BUCKETS, SampledExchange
from .sampled_materialize import SampledMaterializer
from .sampled_residency import Residency


class SampledGroupPlanner:
    def __init__(self, ctx, *, rounds=4, sample_seed=20261026):
        from .planning import allocate_planning_outputs

        self.ctx = ctx
        self.rank = int(ctx["rank"])
        self.group = ctx.get("group")
        self.rounds = rounds
        self.sample_seed = sample_seed
        R, S, K, E, B = (int(ctx[k]) for k in ("R", "S", "K", "E", "B"))
        if (R, E, K) != (8, 352, 8) or S < 32 or not 1 <= B <= 44:
            raise ValueError("sampled_group requires R=8, E=352, K=8, S>=32, 1<=B<=44")
        device = ctx["meta_buf"].device
        self.routes = torch.empty((S, K), device=device, dtype=torch.int32)
        self.seed_owner = torch.empty_like(self.routes)
        self.seed, self.seed_cu = allocate_planning_outputs(ctx)
        self.residency = Residency(ctx, sample_seed=sample_seed)
        self.support = self.residency.support
        self.exchange = SampledExchange(
            self.routes, self.support, source=self.rank, rounds=rounds,
        )
        self.owner = self.exchange.owner
        self.materializer = SampledMaterializer(ctx)
        self.gathered_counts = torch.empty((R, BUCKETS), device=device, dtype=torch.int32)
        self.status = torch.zeros(1, device=device, dtype=torch.int32)
        self.valid = torch.empty(1, device=device, dtype=torch.bool)
        # Explicit device-side rejection hook for fallback correctness tests.
        self.force_reject = torch.zeros_like(self.status)

    def __call__(self, topk, tpe, cu, plan):
        return self.run(topk, tpe, cu, plan)

    def run(self, topk, tpe, cu, plan):
        from .planning import launch_moon_planning

        self.routes.copy_(topk.view_as(self.routes))
        launch_moon_planning(self.ctx, topk, tpe, self.seed_cu, self.seed)
        self.materializer.decode_owner(self.seed.dst, self.seed_owner)
        self.residency.run(self.routes, self.seed_owner, self.seed.experts_to_copy)
        self.materializer.prepare_support(self.support)
        self.exchange.prepare_support()
        self.exchange.reset(self.seed_owner)
        for step in range(self.rounds):
            local_counts = self.exchange.propose(step)
            dist.all_gather_into_tensor(
                self.gathered_counts.view(-1), local_counts, group=self.group,
            )
            self.exchange.commit(step, self.gathered_counts)
        candidate, candidate_cu = self.materializer.materialize(
            self.routes, self.owner, self.seed_owner,
        )
        torch.maximum(self.residency.status, self.materializer.status, out=self.status)
        torch.maximum(self.status, self.force_reject, out=self.status)
        dist.all_reduce(self.status, op=dist.ReduceOp.MAX, group=self.group)
        torch.eq(self.status, 0, out=self.valid)
        torch.where(self.valid, candidate_cu, self.seed_cu, out=cu)
        for name in ("dst", "experts_to_copy", "zero_fill_ranges", "remote_stats"):
            torch.where(
                self.valid, getattr(candidate, name), getattr(self.seed, name),
                out=getattr(plan, name),
            )
        # Fresh native dispatch builds plan-owned dedup data from these final
        # addresses and source records. Never copy candidate scratch dedup data.
        self.materializer.publish(plan.dst)
