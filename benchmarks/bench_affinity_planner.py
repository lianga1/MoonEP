"""Integrated MoonEP vs affinity_gpu planner benchmark (metadata collectives included).

Example: torchrun --standalone --nproc_per_node=8 -m benchmarks.bench_affinity_planner
Replay: add --replay /path/to/routes.npz (key: routes, shape [R,S,K]).
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from benchmarks.bench_comm import setup, time_gpu_op
from moonep.planning import (
    allocate_planning_outputs,
    launch_moon_planning,
    launch_planning,
)
from tests.affinity_utils import check_public_dispatch, gather_raw, plan_metrics
from tests.kernel_test_utils import KernelCase, planning_invariant_errors

from moonep import Buffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, default=100000, help="global tokens, divisible by EP size"
    )
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, default=Path("affinity_results.json"))
    args = parser.parse_args()
    rank, R, _ = setup()
    buffer = None
    try:
        E, K = args.experts, args.topk
        if args.replay:
            with np.load(args.replay, allow_pickle=False) as data:
                original = data["routes"].copy()
            if original.ndim != 3 or original.shape[0] != R or original.shape[2] != K:
                raise ValueError(
                    "replay must have shape [physical EP ranks, S, topk]; no implicit repartition"
                )
            if not np.issubdtype(original.dtype, np.integer):
                raise ValueError("replay expert IDs must be integers")
            routes = torch.from_numpy(original.astype(np.int64))
            source = "replay"
        else:
            if args.tokens % R:
                raise ValueError("--tokens must be divisible by world size")
            # Discrete bell-shaped popularity; endpoint ratio about 7:1.
            # Weighted top-k without replacement preserves distinct expert IDs.
            index = torch.arange(E, dtype=torch.float64)
            sigma = ((E - 1) / 2) / (2 * np.log(7)) ** 0.5
            popularity = torch.exp(-0.5 * ((index - (E - 1) / 2) / sigma) ** 2)
            generator = torch.Generator().manual_seed(20261010)
            batches = [
                torch.multinomial(
                    popularity.expand(min(2048, args.tokens - i), -1),
                    K,
                    replacement=False,
                    generator=generator,
                )
                for i in range(0, args.tokens, 2048)
            ]
            routes = torch.cat(batches).reshape(R, -1, K)
            source = "synthetic_gaussian_7to1"
        if not bool(((routes >= 0) & (routes < E)).all()):
            raise ValueError("expert IDs outside [0,experts)")
        if bool((routes.sort(dim=-1).values.diff(dim=-1) == 0).any()):
            raise ValueError("duplicate expert within a token")
        S = routes.shape[1]
        buffer = Buffer(
            S,
            128,
            K,
            E,
            R,
            B=args.slots,
            planning_backend="affinity_gpu",
            planning_sweeps=args.sweeps,
            explicitly_destroy=True,
        )
        ctx = buffer._require_ctx()
        topk = routes[rank].to(device="cuda", dtype=torch.int32).contiguous()
        tpe = torch.bincount(topk.flatten().long(), minlength=E).int()
        seed, seed_cu = allocate_planning_outputs(ctx)
        optimized, optimized_cu = allocate_planning_outputs(ctx)

        def baseline():
            launch_moon_planning(ctx, topk.flatten(), tpe, seed_cu, seed)

        def affinity():
            launch_planning(ctx, topk.flatten(), tpe, optimized_cu, optimized)

        baseline()
        baseline_metrics = plan_metrics(gather_raw(seed), routes, ctx["NvS"], E)
        # The public API, rather than monkeypatching a benchmark's launch function.
        plan, cu = check_public_dispatch(
            buffer, topk, tpe, async_finish=True, reuse=True
        )
        case = KernelCase("benchmark", S, K, E // R, 128, 32, B=args.slots)
        errors = planning_invariant_errors(
            case, ctx, plan.dst, cu, plan.experts_to_copy
        )
        assert not errors, errors
        base_times = [
            time_gpu_op(baseline, 3, args.iters, dist.group.WORLD)
            for _ in range(args.repeats)
        ]
        ours_times = [
            time_gpu_op(affinity, 3, args.iters, dist.group.WORLD)
            for _ in range(args.repeats)
        ]
        final = plan_metrics(gather_raw(optimized), routes, ctx["NvS"], E)
        status = int(ctx["affinity_gpu_planner"].gpu.status.item())
        assert (
            final["remote_token_destinations"]
            <= baseline_metrics["remote_token_destinations"]
        )
        assert max(final["active_replicas_per_rank"]) <= args.slots
        if status == 0:
            assert final["loads"] == [S * K] * R
        record = {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "source": source,
            "replay_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest()
            if args.replay
            else None,
            "synthetic_seed": None if args.replay else 20261010,
            "physical_ranks": R,
            "global_tokens": R * S,
            "experts": E,
            "topk": K,
            "slots": args.slots,
            "sweeps": args.sweeps,
            "baseline": baseline_metrics,
            "affinity": final,
            "optimizer_status": status,
            "fallback_to_moon": status != 0,
            "official_us": base_times,
            "affinity_us": ours_times,
            "official_median_us": statistics.median(base_times),
            "affinity_median_us": statistics.median(ours_times),
            "communication_reduction": 1
            - final["remote_token_destinations"]
            / baseline_metrics["remote_token_destinations"]
            if baseline_metrics["remote_token_destinations"]
            else 0,
            "timer": "upstream bench_comm.time_gpu_op; CUDA Graph; cross-rank mean",
            "includes_metadata_collectives": True,
            "public_dispatch_combine_reuse_passed": True,
            "planning_invariants_passed": True,
        }
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record, indent=2), flush=True)
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
