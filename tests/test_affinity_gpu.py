"""Run: torchrun --standalone --nproc_per_node=8 -m pytest -q tests/test_affinity_gpu.py"""

import pytest
import torch
import torch.distributed as dist
from moonep.planning import allocate_planning_outputs, launch_planning

from moonep import Buffer
from tests.affinity_utils import check_public_dispatch, gather_raw, plan_metrics
from tests.kernel_test_utils import KernelCase, make_topk, planning_invariant_errors


@pytest.mark.parametrize(
    "sweeps,slots,padding,async_finish",
    [
        (0, 32, 1, False),
        (2, 32, 128, False),
        (2, 48, 128, True),
    ],
)
def test_public_affinity(dist_env, sweeps, slots, padding, async_finish):
    rank, R = dist_env
    if R not in (2, 4, 8):
        pytest.skip("test requires 2, 4 or 8 GPUs")
    case = KernelCase(
        "affinity",
        257,
        8,
        256 // R,
        128,
        32,
        B=slots,
        token_padding=padding,
        routing="biased",
        bias_ratio=1.0,
    )
    topk, tpe = make_topk(case, rank, R)
    with_buffer = Buffer(
        case.S,
        case.H,
        case.K,
        256,
        R,
        B=slots,
        token_padding=padding,
        planning_backend="affinity_gpu",
        planning_sweeps=sweeps,
        explicitly_destroy=True,
    )
    try:
        ctx = with_buffer._require_ctx()
        plan, cu = check_public_dispatch(
            with_buffer, topk, tpe, async_finish=async_finish, reuse=True
        )
        assert not planning_invariant_errors(
            case, ctx, plan.dst, cu, plan.experts_to_copy
        )
        adapter = ctx["affinity_gpu_planner"]
        routes = adapter.routes.cpu().long()
        initial = plan_metrics(adapter.seed_raw.cpu(), routes, ctx["NvS"], 256)
        final = plan_metrics(gather_raw(plan), routes, ctx["NvS"], 256)
        assert (
            final["remote_token_destinations"] <= initial["remote_token_destinations"]
        )
        assert max(final["active_replicas_per_rank"]) <= slots
        if slots >= 256 // R:
            assert int(adapter.gpu.status.item()) == 0, "unexpected optimizer fallback"
        if int(adapter.gpu.status.item()) == 0:
            assert final["loads"] == [case.S * case.K] * R
        else:
            assert torch.equal(plan.dst, adapter.seed.dst)
        # Capture the integrated launch path and replay with reused buffers.
        captured_plan, captured_cu = allocate_planning_outputs(ctx)
        for _ in range(3):
            launch_planning(ctx, topk.flatten(), tpe, captured_cu, captured_plan)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_planning(ctx, topk.flatten(), tpe, captured_cu, captured_plan)
        graph.replay()
        torch.cuda.synchronize()
        assert not planning_invariant_errors(
            case, ctx, captured_plan.dst, captured_cu, captured_plan.experts_to_copy
        )
    finally:
        with_buffer.destroy()


def test_subgroup_and_fallback(dist_env):
    rank, R = dist_env
    if R < 4 or R % 2:
        pytest.skip("requires even world size >=4")
    # Nonzero global root tests group-local source rank translation.
    groups = [dist.new_group(list(range(i, i + 2))) for i in range(0, R, 2)]
    group = groups[rank // 2]
    buffer = Buffer(
        33,
        128,
        4,
        32,
        2,
        B=16,
        group=group,
        planning_backend="affinity_gpu",
        explicitly_destroy=True,
    )
    try:
        ctx = buffer._require_ctx()
        local_rank = dist.get_rank(group)
        topk = (
            (
                torch.arange(33, device="cuda")[:, None]
                + torch.arange(4, device="cuda")[None, :]
                + local_rank
            )
            % 32
        ).int()
        tpe = torch.bincount(topk.flatten().long(), minlength=32).int()
        check_public_dispatch(buffer, topk, tpe, async_finish=True, reuse=True)
        # Deliberately reject optimization to verify the GPU-only fallback
        # publishes the original native dst/src_info and still executes correctly.
        adapter = ctx["affinity_gpu_planner"]
        optimize = adapter.gpu.optimize

        def rejected(routes):
            optimize(routes)
            adapter.gpu.status.fill_(1)

        adapter.gpu.optimize = rejected
        plan, _ = check_public_dispatch(buffer, topk, tpe, reuse=True)
        assert torch.equal(plan.dst, adapter.seed.dst)
    finally:
        buffer.destroy()
        for item in groups:
            if item != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(item)
