"""Functional check for the native planner's fused physical route output.

Run on one H200 node with two GPUs, for example:
    torchrun --nproc_per_node=2 -m pytest -s tests/test_physical_topk.py
"""

import torch
import torch.distributed as dist
import pytest

from tests.kernel_test_utils import (
    KernelCase,
    assert_tensor_equal_all_ranks,
    init_case,
    make_topk,
)


def test_native_physical_topk_matches_dst_oracle(dist_env):
    rank, R = dist_env
    if R != 2:
        pytest.skip(f"physical-topk functional test requires 2 ranks, got {R}")

    # Both source ranks route to expert 0.  The planner must place part of
    # that overloaded home group on the other rank, exercising a remote copy
    # slot as well as duplicate top-k entries.
    case = KernelCase(
        "physical_topk_remote",
        S=4,
        K=2,
        epn=3,
        H=16,
        num_sms=1,
        B=2,
        token_padding=8,
        routing="single_expert",
    )
    ctx = init_case(case, R)
    topk, tpe = make_topk(case, rank, R)

    from moonep.planning import allocate_planning_outputs, launch_planning

    plan, cu_seqlens = allocate_planning_outputs(ctx)
    launch_planning(
        ctx,
        topk.reshape(-1).contiguous(),
        tpe,
        cu_seqlens,
        plan,
    )

    all_cu = [torch.empty_like(cu_seqlens) for _ in range(R)]
    dist.all_gather(all_cu, cu_seqlens)
    all_cu_cpu = torch.stack(all_cu).cpu()
    dst_cpu = plan.dst.cpu()
    topk_cpu = topk.reshape(-1).cpu()
    etc_cpu = plan.experts_to_copy.cpu()

    E = int(ctx["E"])
    B = int(ctx["B"])
    epn = E // R
    nv_s = int(ctx["NvS"])
    expected = torch.empty_like(plan.physical_topk.cpu())
    for i in range(expected.numel()):
        raw_dst = int(dst_cpu[i])
        if raw_dst < 0:
            raw_dst = -raw_dst - 1
        dest = raw_dst // nv_s
        offset = raw_dst % nv_s
        group = int(torch.searchsorted(all_cu_cpu[dest], offset, right=True))
        if group < E:
            slot = group - dest * epn
        else:
            slot = epn + group - E
        expected[i] = dest * (epn + B) + slot

        physical_id = int(expected[i])
        assert 0 <= physical_id < R * (epn + B)
        assert dest == physical_id // (epn + B)
        physical_slot = physical_id % (epn + B)
        logical_expert = int(topk_cpu[i])
        owner = logical_expert // epn
        if dest == owner:
            assert physical_slot == logical_expert % epn
        else:
            assert physical_slot >= epn
            assert physical_slot - epn < B
            assert int(etc_cpu[dest, physical_slot - epn]) == logical_expert

    assert_tensor_equal_all_ranks(
        "physical_topk",
        plan.physical_topk,
        expected.to(plan.physical_topk.device),
        rank,
        R,
    )
    # Confirm that this case actually exercised the remote branch on every
    # rank, rather than passing with home experts only.
    has_remote = any(
        int(plan.physical_topk[i].item()) // (epn + B) != int(topk_cpu[i]) // epn
        for i in range(expected.numel())
    )
    remote_flag = torch.tensor([int(has_remote)], dtype=torch.int32, device="cuda")
    all_remote = [torch.empty_like(remote_flag) for _ in range(R)]
    dist.all_gather(all_remote, remote_flag)
    assert sum(int(flag.item()) for flag in all_remote) > 0
