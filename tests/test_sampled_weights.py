"""Actual eight-GPU sampled-plan weight/parameter-gradient lifecycle test.

Run:
  torchrun --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_weights.py

Public dispatch produces the plan. Public prefetch_weight/reduce_grad consume
that SAME plan. Every source/home and replica slot is physical on its owner;
the full and reduction views are genuine VMM peer mappings, not copied host
references or full replicated tensors. Two consecutive disjoint hot-expert
sets change the copy map and refresh a separately encoded weight version.
No GEMM, optimizer, latency claim, or complete model training is implied.
"""

from dataclasses import dataclass
import os
import socket

import pytest
import torch
import torch.distributed as dist

from moonep import Buffer
from moonep.buffer import _all_gather_shareables, _exchange_ipc_fds, _use_fabric_for_group


R, E, K, S, H, B = 8, 352, 8, 257, 256, 44
NAMES = ("gate", "up", "down")
SENTINEL = -123


def _assert_all(ok, label):
    value = torch.tensor([int(bool(ok))], device="cuda", dtype=torch.int32)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    assert int(value.item()) == 1, label


def _ready():
    """All producer writes finish before any peer starts one-sided reads."""
    torch.cuda.synchronize()
    dist.barrier()


def _wait(event):
    if event is not None:
        torch.cuda.current_stream().wait_event(event)
    torch.cuda.synchronize()


def _compile_all(buffer, intermediate):
    """Compile every used variant before a native device barrier can launch."""
    from moonep import (planning, inter_rank_sync, dispatch, combine,
                        dispatch_epilogue, combine_prologue, prefetch, grad_reduce)

    c = buffer._require_ctx()
    nvs, nvsp, stride, sms, device = (int(c[k]) for k in
        ("NvS", "NvS_padded", "meta_chunk_padded", "num_sms", "device"))
    planning._get_compiled(R, E, B, S, K, int(c["NvS_capacity"]), nvs,
        int(c["num_vblocks"]), stride, int(c["TPE_OFF"]), int(c["PLAN_OFF"]),
        int(c["BARRIER_OFF"]), int(c["TOPK0_OFF"]), int(c["ORDER_OFF"]),
        int(c["ORDER0_OFF"]), int(c["token_padding"]), sms)
    inter_rank_sync._get_compiled(R, stride, inter_rank_sync._num_threads_for_ranks(R), device)
    dispatch._get_compiled(H, R, S, K, E+B, nvs, nvsp, stride,
        int(c["SRC_INFO_OFF"]), sms, True, True, device, False)
    combine._get_compiled(H, R, S, K, nvs, nvsp, stride, sms, False, device, False)
    dispatch_epilogue._get_compiled(H, nvs, int(c["num_sms_dedup"]), device, False)
    combine_prologue._get_compiled(H, R, nvs, int(c["num_sms_dedup"]), device, False)
    for h, hp in ((intermediate, H), (H, intermediate)):
        prefetch._get_compiled(E, h, hp, B, sms, device, torch.bfloat16)
        grad_reduce._get_compiled(E, h, hp, R, B, stride, sms, device)
    _ready()


@dataclass
class Projection:
    name: str
    index: int
    home: torch.Tensor
    slots: torch.Tensor
    full: torch.Tensor
    reduce: torch.Tensor | None


def _projection(rank, name, index, h, hp, dtype, fabric):
    """Parent API has equal-chunk nvl_dist_map, not map_segments.

    B=E/R=44 makes all chunks the same size. Eight home chunks plus this
    rank's slot chunk form [E+B,h,hp]; eight slot chunks form [R,B,h,hp].
    Both views import the SAME slot handle. No new C++ mapping API is used.
    """
    from moonep._C import nvl_dist_alloc, nvl_dist_map, nvl_release_mem_handle

    home, home_handle, home_owned = nvl_dist_alloc([B, h, hp], dtype, fabric)
    slots, slot_handle, slot_owned = nvl_dist_alloc([B, h, hp], dtype, fabric)
    nvl_release_mem_handle(home_owned)
    nvl_release_mem_handle(slot_owned)
    imported_fds = []
    try:
        if fabric:
            homes = _all_gather_shareables(home_handle, None)
            all_slots = _all_gather_shareables(slot_handle, None)
        else:
            home_fd, slot_fd = int(home_handle.item()), int(slot_handle.item())
            try:
                received = _exchange_ipc_fds(home_fd, list(range(R)), rank, R, None)
            finally:
                os.close(home_fd)
            imported_fds.extend(received.values())
            try:
                received_slots = _exchange_ipc_fds(slot_fd, list(range(R)), rank, R, None)
            finally:
                os.close(slot_fd)
            imported_fds.extend(received_slots.values())
            homes = torch.tensor([received[r] for r in range(R)], dtype=torch.int64)
            all_slots = torch.tensor([received_slots[r] for r in range(R)], dtype=torch.int64)
        handles = torch.cat((homes, all_slots[rank:rank+1]), dim=0)
        full = nvl_dist_map([B, h, hp], dtype, handles, rank, R+1, fabric)
        reduction = None
        if dtype == torch.float32:
            reduction = nvl_dist_map([B, h, hp], dtype, all_slots, rank, R, fabric).view(R, B, h, hp)
        return Projection(name, index, home, slots, full, reduction)
    finally:
        for fd in imported_fds:
            os.close(fd)


def _weight_values(experts, h, hp, projection, version):
    experts = torch.as_tensor(experts, device="cuda", dtype=torch.int32)
    rows = torch.arange(h, device="cuda", dtype=torch.int32)[None, :, None]
    cols = torch.arange(hp, device="cuda", dtype=torch.int32)[None, None, :]
    values = ((experts[:, None, None] % 31 + 3*rows + 5*cols
               + projection*29 + version*47) % 127 + 1).to(torch.bfloat16)
    # The quotient plus the mod-31 body identifies every E=352 expert.
    # All values are small exact BF16 integers; version/projection cannot alias.
    values[:, :, 0] = (experts // 31 + 1)[:, None].to(torch.bfloat16)
    values[:, :, 1] = version*8 + projection + 1
    return values


def _home_values(experts, h, hp, projection, version):
    experts = torch.as_tensor(experts, device="cuda", dtype=torch.float32)
    rows = torch.arange(h, device="cuda", dtype=torch.float32)[None, :, None] % 17
    cols = torch.arange(hp, device="cuda", dtype=torch.float32)[None, None, :] % 11
    return (experts[:, None, None]*17 + projection*1009 + version*19 + rows*3 + cols).contiguous()


def _slot_values(source, slots, h, hp, projection, version):
    slots = torch.as_tensor(slots, device="cuda", dtype=torch.float32)
    rows = torch.arange(h, device="cuda", dtype=torch.float32)[None, :, None] % 13
    cols = torch.arange(hp, device="cuda", dtype=torch.float32)[None, None, :] % 7
    return ((source+1)*101 + slots[:, None, None]*11 + projection*1237
            + version*23 + rows*2 + cols).contiguous()


def _refresh(weights, gradients, rank, version):
    local_experts = torch.arange(rank*B, (rank+1)*B, device="cuda")
    local_slots = torch.arange(B, device="cuda")
    for p in weights:
        h, hp = p.home.shape[1:]
        p.home.copy_(_weight_values(local_experts, h, hp, p.index, version))
        p.slots.fill_(SENTINEL)
    for p in gradients:
        h, hp = p.home.shape[1:]
        p.home.copy_(_home_values(local_experts, h, hp, p.index, version))
        # Deliberately initialize inactive slots too; the reduction must leave
        # them untouched and clear only the slots selected by this plan.
        p.slots.copy_(_slot_values(rank, local_slots, h, hp, p.index, version))
    _ready()


def _check_weights(weights, mapping, rank, version):
    local = mapping[rank]
    active = torch.tensor([b for b, e in enumerate(local) if e >= 0], device="cuda", dtype=torch.long)
    inactive = torch.tensor([b for b, e in enumerate(local) if e < 0], device="cuda", dtype=torch.long)
    experts = [e for e in local if e >= 0]
    ok = True
    for p in weights:
        h, hp = p.home.shape[1:]
        expected = _weight_values(experts, h, hp, p.index, version)
        ok &= torch.equal(p.slots[active], expected)
        ok &= torch.equal(p.full[E:][active], expected)
        ok &= torch.equal(p.slots[inactive], torch.full_like(p.slots[inactive], SENTINEL))
        ok &= torch.equal(p.full[rank*B:(rank+1)*B],
                          _weight_values(range(rank*B, (rank+1)*B), h, hp, p.index, version))
    _assert_all(ok, f"weight version {version}: peer copies, inactive slots, or home contents differ")


def _check_gradients(gradients, mapping, rank, version):
    _ready()  # Every source has finished its local consumed-slot clear.
    active = [b for b, e in enumerate(mapping[rank]) if e >= 0]
    ok = True
    for p in gradients:
        h, hp = p.home.shape[1:]
        expected = _home_values(range(rank*B, (rank+1)*B), h, hp, p.index, version)
        for source in range(R):
            additions = _slot_values(source, range(B), h, hp, p.index, version)
            for b, e in enumerate(mapping[source]):
                if rank*B <= e < (rank+1)*B:
                    expected[e-rank*B].add_(additions[b])
        # Entire home tensor is checked: this also covers experts that have
        # no replicas and therefore must retain their original local gradient.
        ok &= torch.equal(p.home, expected)
        ok &= torch.equal(p.full[rank*B:(rank+1)*B], expected)
        expected_slots = _slot_values(rank, range(B), h, hp, p.index, version)
        expected_slots[active] = 0
        ok &= torch.equal(p.slots, expected_slots)
        ok &= torch.equal(p.full[E:], expected_slots)
        ok &= torch.equal(p.reduce[rank], expected_slots)
    _assert_all(ok, f"gradient version {version}: home+remote sum, slot alias/clear, or inactive preservation differs")


def test_sampled_plan_weight_and_parameter_gradient_lifecycle(dist_env):
    rank, world = dist_env
    if world != R:
        pytest.skip("requires eight actual GPUs on one NVLink host")
    assert int(os.environ.get("LOCAL_WORLD_SIZE", 0)) == R
    assert torch.cuda.device_count() >= R
    identities = [None]*R
    dist.all_gather_object(identities, (socket.gethostname(), torch.cuda.current_device()))
    assert len(set(identities)) == R, "ranks must not share a physical GPU"

    from moonep._C import get_vmm_granularity
    granularity = get_vmm_granularity()
    granularities = [None]*R
    dist.all_gather_object(granularities, granularity)
    assert len(set(granularities)) == 1, "this test requires homogeneous VMM alignment"
    intermediate = 128
    while B*H*intermediate*2 % granularity:
        intermediate *= 2
    shapes = ((intermediate, H), (intermediate, H), (H, intermediate))
    buffer = Buffer(S, H, K, E, R, B=B, token_padding=128, num_sms=32,
                    planning_backend="sampled_group", planning_rounds=4,
                    planning_sample_seed=20261026, enable_pdl=False, explicitly_destroy=True)
    weights, gradients = [], []
    try:
        _compile_all(buffer, intermediate)
        fabric = _use_fabric_for_group(None)
        for i, (name, (h, hp)) in enumerate(zip(NAMES, shapes)):
            weights.append(_projection(rank, name, i, h, hp, torch.bfloat16, fabric))
            gradients.append(_projection(rank, name, i, h, hp, torch.float32, fabric))
        previous = None
        for version, expert_home in enumerate((1, 5), start=1):
            asynchronous = version == 2
            # All sources target the same eight experts in a different home
            # group each round: balancing requires real active remote copies.
            topk = (expert_home*B + torch.arange(K, device="cuda", dtype=torch.int32)).expand(S, K).contiguous()
            tpe = torch.bincount(topk.flatten().long(), minlength=E).int()
            x = ((torch.arange(S, device="cuda") % 11 + rank*13 + 1)
                 .to(torch.bfloat16)[:, None].expand(S, H).contiguous())
            router = torch.ones((S, K), dtype=torch.float32, device="cuda")
            _refresh(weights, gradients, rank, version)
            result = buffer.dispatch(x, router, topk, tpe, async_finish=asynchronous)
            if asynchronous:
                _wait(result[-1])
            else:
                torch.cuda.synchronize()
            dispatched, _, cu, plan = result[:4]
            _assert_all(int(buffer._require_ctx()["sampled_group_planner"].status.item()) == 0,
                        "expected accepted sampled plan, not native fallback")
            mapping = plan.experts_to_copy.cpu().tolist()
            gathered = [None]*R
            dist.all_gather_object(gathered, mapping)
            _assert_all(all(item == mapping for item in gathered), "copy table differs between ranks")
            live = sum(e >= 0 for row in mapping for e in row)
            _assert_all(live > 0 and all(e == -1 or (0 <= e < E and e//B != source)
                        for source, row in enumerate(mapping) for e in row), "expected real remote expert copies")
            if previous is not None:
                _assert_all(mapping != previous, "the two routes must change the expert-copy plan")
            previous = mapping
            starts = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), cu[:-1]))
            live_copy_rows = (cu[E:] - starts[E:]).sum().to(torch.int64)
            dist.all_reduce(live_copy_rows, op=dist.ReduceOp.SUM)
            _assert_all(int(live_copy_rows.item()) > 0, "copy slots must carry actual execution rows")
            event = buffer.prefetch_weight(plan=plan, async_finish=asynchronous,
                **{f"full_{p.name}_weight": p.full for p in weights})
            _wait(event)
            _check_weights(weights, mapping, rank, version)
            # Real public combine, with identity expert outputs, proves that
            # this plan has also been consumed by the native token path.
            combined = buffer.combine(plan=plan, hidden_nvsh=dispatched, async_finish=asynchronous)
            if asynchronous:
                _wait(combined[-1])
            else:
                torch.cuda.synchronize()
            _assert_all(torch.equal(combined[0], x*K), "identity dispatch/combine payload mismatch")
            # grad_reduce has no entry ready barrier. Establish global producer
            # completion explicitly; the public async API then waits its local
            # producer stream and serializes Buffer's metadata resources.
            _ready()
            grad_args = {f"full_{p.name}_grad": p.full for p in gradients}
            grad_args.update({f"{p.name}_reduce_buffer": p.reduce for p in gradients})
            event = buffer.reduce_grad(plan=plan, async_finish=asynchronous, **grad_args)
            _wait(event)
            _check_gradients(gradients, mapping, rank, version)
            if rank == 0:
                print(f"[PASS] sampled_group weights/FP32 grads version={version}, copies={live}, "
                      f"H={H}, I={intermediate}, async={asynchronous}", flush=True)
        _ready()
    finally:
        # This is one forward/backward lifecycle at a time; no old version or
        # plan remains in flight when the next refresh reuses its slots.
        torch.cuda.synchronize()
        gradients.clear()
        weights.clear()
        buffer.destroy()
