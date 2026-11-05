"""Planner-only variable-length route test.

Run with:
    torchrun --nproc_per_node=2 -m pytest -s tests/test_dynamic_planning.py
"""

import pytest
import torch
import torch.distributed as dist


def test_planner_only_reuses_capacity_for_equal_rank_lengths(dist_env):
    rank, world_size = dist_env
    if world_size != 2:
        pytest.skip(
            "dynamic planner test requires two EP ranks, "
            f"got {world_size}"
        )

    from moonep import Buffer

    # ``Buffer.S`` is the maximum sequence length *per EP rank*.  Express the
    # workload in global sequence lengths so the test matches the integration
    # contract: every rank receives the same local length on each call.
    max_global_seq_len = 2048
    assert max_global_seq_len % world_size == 0
    max_tokens = max_global_seq_len // world_size
    K = 2
    E = 6
    B = 2
    buffer = Buffer(
        max_tokens,
        16,
        K,
        E,
        world_size,
        B=B,
        num_sms=1,
        token_padding=8,
        planner_only=True,
    )
    try:
        ctx = buffer._require_ctx()
        for global_seq_len in (2048, 1024, 512):
            assert global_seq_len % world_size == 0
            current_tokens = global_seq_len // world_size
            token_ids = torch.arange(
                current_tokens, dtype=torch.int32, device="cuda"
            ) + rank * current_tokens
            topk = torch.stack(
                (token_ids.remainder(E), (token_ids + 1).remainder(E)),
                dim=1,
            ).contiguous()
            tokens_per_expert = torch.bincount(
                topk.reshape(-1),
                minlength=E,
            ).to(torch.int32)
            plan, cu_seqlens = buffer.plan(topk, tokens_per_expert)

            assert plan.N == current_tokens * K
            assert plan.N_capacity == max_tokens * K
            assert tuple(plan.physical_topk.shape) == (current_tokens * K,)
            assert tuple(plan.dst.shape) == (current_tokens * K,)
            assert tuple(cu_seqlens.shape) == (E + B,)
            assert int(cu_seqlens.max().item()) <= int(ctx["NvS"])
            assert bool(
                ((plan.physical_topk >= 0)
                 & (plan.physical_topk < world_size * (E // world_size + B))).all()
            )

            # Both physical_topk/dst and cu_seqlens are rank-local outputs:
            # each rank owns a different source-token range and may receive a
            # different balanced destination slice.  Validate their local
            # shapes/ranges instead of requiring cross-rank element equality.
            assert bool((cu_seqlens[1:] >= cu_seqlens[:-1]).all())
            dist.barrier()
    finally:
        if not buffer.destroyed:
            buffer.destroy()
