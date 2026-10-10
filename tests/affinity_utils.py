"""Independent CPU audit helpers; deliberately outside timed GPU regions."""

import torch
import torch.distributed as dist


def gather_raw(plan, group=None):
    tensors = [torch.empty_like(plan.dst) for _ in range(dist.get_world_size(group))]
    dist.all_gather(tensors, plan.dst, group=group)
    return torch.stack(tensors).cpu()


def plan_metrics(raw, routes, nvs, experts):
    """Count unique remote token destinations, explicit pair load and replicas."""
    R, S, K = routes.shape
    raw = raw.reshape(R, S, K)
    addresses = torch.where(raw < 0, -raw - 1, raw).long()
    dst = addresses // nvs
    assert bool(((dst >= 0) & (dst < R)).all())
    assert addresses.unique().numel() == routes.numel(), "duplicate execution row"
    pairs = torch.bincount(
        (dst * experts + routes).flatten(), minlength=R * experts
    ).view(R, experts)
    visits = torch.zeros(R, S, R, dtype=torch.bool)
    visits.scatter_(2, dst, True)
    sources = torch.arange(R)[:, None, None]
    remote = int((visits & (torch.arange(R)[None, None, :] != sources)).sum())
    all_destinations = int(visits.sum())
    seen = torch.zeros(R, S, R, dtype=torch.bool)
    for k in range(K):
        d = dst[:, :, k : k + 1]
        assert torch.equal(raw[:, :, k : k + 1] < 0, seen.gather(2, d)), (
            "bad duplicate marker"
        )
        seen.scatter_(2, d, True)
    replicas = (
        (pairs > 0)
        & (torch.arange(experts)[None, :] // (experts // R) != torch.arange(R)[:, None])
    ).sum(1)
    loads = pairs.sum(1)
    return {
        "remote_token_destinations": remote,
        "all_token_destinations": all_destinations,
        "N": routes.numel() / all_destinations,
        "loads": loads.tolist(),
        "load_spread_fraction": float(
            (loads.max() - loads.min()) / loads.float().mean()
        ),
        "active_replicas_per_rank": replicas.tolist(),
    }


def check_expert_rows(plan, cu, routes, ctx):
    """Independently map every execution row back to its assigned expert."""
    R, S, K = routes.shape
    E, B, nvs = (int(ctx[k]) for k in ("E", "B", "NvS"))
    raw = gather_raw(plan, ctx.get("group")).reshape(R, S, K)
    address = torch.where(raw < 0, -raw - 1, raw).long()
    dst, offset = address // nvs, address % nvs
    copies = plan.experts_to_copy.cpu().long()
    gathered_cu = [torch.empty_like(cu) for _ in range(R)]
    dist.all_gather(gathered_cu, cu, group=ctx.get("group"))
    ends = torch.stack(gathered_cu).cpu().long()
    starts = torch.cat((torch.zeros(R, 1, dtype=torch.long), ends[:, :-1]), dim=1)
    groups = torch.full((R, E), -1, dtype=torch.long)
    for r in range(R):
        home = torch.arange(r * (E // R), (r + 1) * (E // R))
        groups[r, home] = home
        active = copies[r] >= 0
        assert copies[r, active].unique().numel() == int(active.sum())
        assert bool((copies[r, active] // (E // R) != r).all())
        groups[r, copies[r, active]] = E + torch.arange(B)[active]
    assigned_group = groups[dst, routes]
    assert bool((assigned_group >= 0).all()), "expert absent from home/copy layout"
    row_start = starts[dst, assigned_group]
    row_end = ends[dst, assigned_group]
    assert bool(((offset >= row_start) & (offset < row_end)).all()), (
        "wrong expert execution segment"
    )
    counts = torch.bincount((dst * E + routes).flatten(), minlength=R * E).view(R, E)
    # No holes or assignments placed inside padding; uniqueness is checked above.
    assert bool((offset < row_start + counts[dst, routes]).all()), "pair inside padding"


def check_public_dispatch(buffer, topk, tpe, *, async_finish=False, reuse=False):
    """Exercise public dispatch/combine, including weights and saved-plan reuse."""
    ctx = buffer._require_ctx()
    S, K, H, _R, rank = (int(ctx[k]) for k in ("S", "K", "H", "R", "rank"))
    hidden = (
        ((torch.arange(S, device="cuda") % 7) + rank + 1)
        .to(torch.bfloat16)[:, None]
        .expand(S, H)
        .contiguous()
    )
    weights = (
        torch.arange(S * K, device="cuda", dtype=torch.float32) + rank * S * K
    ).view(S, K)
    result = buffer.dispatch(hidden, weights, topk, tpe, async_finish=async_finish)
    if async_finish:
        torch.cuda.current_stream().wait_event(result[-1])
    dispatched, routed_weights, cu, plan = result[:4]
    group = ctx.get("group")
    if "affinity_gpu_planner" in ctx:
        check_expert_rows(
            plan, cu, ctx["affinity_gpu_planner"].routes.cpu().long(), ctx
        )
    all_raw = gather_raw(plan, group).to("cuda").flatten()
    address = torch.where(all_raw < 0, -all_raw - 1, all_raw).long()
    ids = torch.nonzero(address // ctx["NvS"] == rank).flatten()
    offsets = address[ids] % ctx["NvS"]
    tokens = ids // K
    expected = ((tokens % S) % 7 + tokens // S + 1).to(torch.bfloat16)
    assert torch.equal(dispatched[offsets], expected[:, None].expand(-1, H))
    assert torch.equal(routed_weights[offsets], ids.float())
    # A constant expert output isolates dispatch/combine from FFN arithmetic.
    dispatched.zero_()
    dispatched[offsets] = 1
    combined = buffer.combine(
        plan=plan, hidden_nvsh=dispatched, async_finish=async_finish
    )
    if async_finish:
        torch.cuda.current_stream().wait_event(combined[-1])
    assert torch.equal(combined[0], torch.full_like(combined[0], K))
    if reuse:
        saved = plan.dst.clone()
        reused = buffer.dispatch(hidden, plan=plan)
        assert reused[2] is None
        assert torch.equal(reused[0][offsets], expected[:, None].expand(-1, H))
        assert torch.equal(saved, plan.dst)
    return plan, cu
