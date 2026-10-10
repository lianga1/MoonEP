"""GPU并行交换及原生MoonEP适配。所有路由决策在CUDA kernel中执行。

GPU并行顺序与CPU串行贪心不同：每个rank对一次批量交换，token之间互斥。
默认对角线并发独立rank对，shared驻留位图和CTA队列归约。
预分配buffer、固定扫描次数，无.item()/CPU路由/CPU求解，支持CUDA Graph。
初次编译、初始化及显式audit在计时区外。plan()返回借用的device buffers；
下次调用会覆盖，需要跨微批保留时由调用方clone最终MoonEPCommPlan。
"""

from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load


@lru_cache(None)
def extension():
    return load(
        name="moonep_affinity_gpu_v1",
        sources=[str(Path(__file__).parent / "csrc" / "affinity_planner.cu")],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class GpuSwapPlanner:
    """单GPU持有全局路由元数据，R表示逻辑EP ranks；不等于已经完成分布式集成。"""

    def __init__(
        self,
        shape,
        num_experts=256,
        extra_expert_slots=32,
        sweeps=2,
        device="cuda",
        token_padding=1,
        nvs=None,
        scan_backend="diagonal",
        pack_backend="shared",
        blocks_per_sm=4,
    ):
        self.R, self.S, self.K = map(int, shape)
        self.E, self.B, self.sweeps = num_experts, extra_expert_slots, sweeps
        # Resident argmax encodes frequency and expert-ID tie-break in int32.
        if self.S * self.K > (2**31 - 1024) // 1024:
            raise ValueError("affinity_gpu requires S*K <= 2097151")
        if not (
            2 <= self.R <= 16
            and self.S > 0
            and 1 <= self.K <= min(32, self.E)
            and self.E % self.R == 0
            and self.R <= self.E <= 1024
        ):
            raise ValueError("GPU backend requires 2<=R<=16,K<=32,E<=1024,E%R=0")
        if not (
            0 <= self.B <= self.E - self.E // self.R
            and sweeps >= 0
            and token_padding >= 1
        ):
            raise ValueError("invalid slots, sweeps or padding")
        if scan_backend not in (
            "legacy",
            "persistent",
            "persistent_cta",
            "diagonal",
        ) or pack_backend not in ("legacy", "shared"):
            raise ValueError("invalid CUDA backend")
        if blocks_per_sm not in (1, 2, 4):
            raise ValueError("blocks_per_sm must be 1,2,4")
        self.scan_backend, self.pack_backend = scan_backend, pack_backend
        self.padding = token_padding
        self.nvs = nvs or self.S * self.K + (self.E + self.B) * (token_padding - 1)
        self.ext = extension()

        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=torch.int32)

        self.dst = zeros(self.R, self.S, self.K)
        self.freq = zeros(self.R, self.E)
        self.counts = zeros(self.R, self.E)
        self.support = zeros(self.R, self.E)
        self.status = zeros(1)
        self.stats = zeros(2)
        teams = self.R // 2 if scan_backend == "diagonal" else 1
        self.queues = zeros(4 * teams, self.R * self.S)
        self.sizes = zeros(
            max(8, sweeps * (2 * self.R - 3) * teams * 4)
            if scan_backend == "diagonal"
            else 8
        )
        self.offsets = zeros(self.R, self.E)
        self.cursor = zeros(self.R, self.E)
        self.cu = zeros(self.R, self.E + self.B)
        self.copies = zeros(self.R, self.B)
        self.zeros = zeros(self.R, self.E + self.B, 2)
        self.remote_stats = zeros(self.R, 2)
        self.raw = zeros(self.R, self.S, self.K)
        self.seed_output = zeros(self.R, self.S, self.K)
        self.ordinal = zeros(self.R * self.S * self.K)
        self.hist = zeros((self.R * self.S * self.K + 255) // 256, self.E)
        self.demand = zeros(self.E)
        self.alloc = zeros(self.R, self.E)
        self.blocks = (
            min(
                ((self.R * self.S + 255) // 256) * teams,
                self.ext.persistent_blocks(
                    self.dst.device.index,
                    self.K,
                    self.E,
                    blocks_per_sm,
                    scan_backend == "diagonal",
                ),
            )
            if scan_backend != "legacy"
            else 0
        )
        if scan_backend == "diagonal" and self.blocks < teams:
            raise RuntimeError(
                "device cannot host all diagonal rank-pair teams concurrently"
            )
        # 默认48KiB以上动态shared需显式opt-in；大RE回退已验证通用布局。
        if self.R * self.E * 4 > 48 * 1024:
            self.pack_backend = "legacy"

    def _validate_routes(self, routes):
        # 仅检查host元信息，不读取device值，因此可进入CUDA Graph。
        # 调用方保证expert ID在[0,E)且每token不重复；非法值不允许送入kernel。
        if (
            routes.shape != self.dst.shape
            or routes.dtype != torch.int32
            or routes.device != self.dst.device
            or not routes.is_contiguous()
        ):
            raise ValueError(
                "routes must be contiguous int32 [R,S,K] on planner CUDA device"
            )

    def seed(self, routes):
        """稳定FIFO和rank配额也在GPU执行；输出与原始连续home CPU种子逐对一致。"""
        self._validate_routes(routes)
        self.ext.seed(
            routes,
            self.ordinal,
            self.hist,
            self.demand,
            self.alloc,
            self.seed_output,
            self.E,
        )
        return self.seed_output

    def prepare(self, routes, initial_destinations):
        self._validate_routes(routes)
        if (
            initial_destinations.shape != routes.shape
            or initial_destinations.dtype != torch.int32
            or initial_destinations.device != self.dst.device
        ):
            raise ValueError("initial_destinations must be CUDA int32 [R,S,K]")
        self.dst.copy_(initial_destinations)
        self.freq.zero_()
        self.counts.zero_()
        self.status.zero_()
        self.stats.zero_()
        self.ext.prepare(
            routes,
            self.dst,
            self.freq,
            self.counts,
            self.support,
            self.status,
            self.R,
            self.E,
            self.B,
        )

    def optimize(self, routes):
        if self.sweeps and self.scan_backend != "legacy":
            self.ext.persistent_exchange(
                routes,
                self.dst,
                self.support,
                self.queues,
                self.sizes,
                self.stats,
                self.E,
                self.sweeps,
                self.blocks,
                int(self.scan_backend in ("persistent_cta", "diagonal")),
                self.scan_backend == "diagonal",
            )
            return
        for _ in range(self.sweeps):
            for a in range(self.R):
                for b in range(a + 1, self.R):
                    self.sizes.zero_()
                    self.ext.exchange(
                        routes,
                        self.dst,
                        self.support,
                        self.queues,
                        self.sizes,
                        self.stats,
                        self.E,
                        a,
                        b,
                    )

    def pack(self, routes):
        self.counts.zero_()
        self.cursor.zero_()
        pack = self.ext.pack_shared if self.pack_backend == "shared" else self.ext.pack
        pack(
            routes,
            self.dst,
            self.counts,
            self.offsets,
            self.cu,
            self.copies,
            self.zeros,
            self.remote_stats,
            self.cursor,
            self.raw,
            self.status,
            self.E,
            self.B,
            self.padding,
            self.nvs,
        )

    def plan(self, routes, initial_destinations=None):
        """GPU seed→驻留→正收益交换→原生执行行；返回self作为device结果视图。"""
        if initial_destinations is None:
            initial_destinations = self.seed(routes)
        self.prepare(routes, initial_destinations)
        self.optimize(routes)
        self.pack(routes)
        return self

    def check_status(self):
        """显式同步检查，放在实际dispatch前或离线验证；不可置于CUDA Graph内部。"""
        code = int(self.status.item())
        if code:
            raise RuntimeError(
                f"GPU planner invalid plan: status={code} (1=seed,2=slots,3=rows)"
            )


class AffinityGpuPlanner:
    """R进程GPU路径：官方seed→元数据all-gather→rank0优化/编码→广播→本地src_info。

    包含真实元数据collective，不能把单卡kernel时间冒充整个分布式planner时间。
    rank0决定一次编码，广播结果保证所有rank一致，避免atomic队列顺序分歧。
    """

    def __init__(self, ctx, sweeps=2, **gpu_options):
        from moonep.planning import allocate_planning_outputs

        self.ctx = ctx
        self.rank = int(ctx["rank"])
        self.group = ctx.get("group")
        R, S, K, E, B = (int(ctx[k]) for k in ("R", "S", "K", "E", "B"))
        self.gpu = GpuSwapPlanner(
            (R, S, K),
            E,
            B,
            sweeps,
            torch.cuda.current_device(),
            int(ctx["token_padding"]),
            int(ctx["NvS"]),
            **gpu_options,
        )
        self.seed, self.seed_cu = allocate_planning_outputs(ctx)
        self.routes = torch.empty((R, S, K), device="cuda", dtype=torch.int32)
        self.seed_raw = torch.empty_like(self.routes)
        self.seed_dst = torch.empty_like(self.routes)
        self.valid = torch.empty(1, device="cuda", dtype=torch.bool)
        self.source_rank = (
            dist.get_global_rank(self.group, 0) if self.group is not None else 0
        )

    def __call__(self, topk, tpe, cu, plan):
        from moonep.inter_rank_sync import launch_inter_rank_sync
        from moonep.planning import launch_moon_planning

        ctx = self.ctx
        g = self.gpu
        launch_moon_planning(ctx, topk, tpe, self.seed_cu, self.seed)
        dist.all_gather_into_tensor(
            self.routes.view(-1), topk.view(-1), group=self.group
        )
        dist.all_gather_into_tensor(
            self.seed_raw.view(-1), self.seed.dst, group=self.group
        )
        if self.rank == 0:
            g.ext.decode(self.seed_raw, self.seed_dst, g.nvs)
            g.plan(self.routes, self.seed_dst)
        for tensor in (g.raw, g.cu, g.copies, g.zeros, g.remote_stats, g.status):
            dist.broadcast(tensor, src=self.source_rank, group=self.group)
        # Slot-constrained official seeds can be imbalanced. Keep the official
        # result when the optimizer rejects a seed or the packed row capacity.
        # Device-side selection keeps this fallback safe inside CUDA Graphs.
        torch.eq(g.status, 0, out=self.valid)
        valid = self.valid
        torch.where(valid, g.raw, self.seed_raw, out=g.raw)
        plan.dst.copy_(g.raw[self.rank].view(-1))
        torch.where(valid, g.cu[self.rank], self.seed_cu, out=cu)
        torch.where(
            valid, g.copies, self.seed.experts_to_copy, out=plan.experts_to_copy
        )
        torch.where(
            valid,
            g.zeros[self.rank],
            self.seed.zero_fill_ranges,
            out=plan.zero_fill_ranges,
        )
        torch.where(
            valid,
            g.remote_stats[self.rank],
            self.seed.remote_stats,
            out=plan.remote_stats,
        )
        begin = self.rank * int(ctx["meta_chunk_padded"]) + int(ctx["SRC_INFO_OFF"])
        ctx["meta_buf"][begin : begin + g.nvs].fill_(-1)
        g.ext.publish(
            g.raw,
            ctx["meta_buf"],
            g.S * g.K,
            g.nvs,
            self.rank,
            int(ctx["meta_chunk_padded"]),
            int(ctx["SRC_INFO_OFF"]),
        )
        launch_inter_rank_sync(ctx)
