"""Real DeepEP v1 dispatch/combine + separately labelled NCCL weight prefetch.

Synthetic correctness probe; no real model or performance claim. The repeated
no-CPU-sync dispatch also exercises capacity buffers and invalid tail rows.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import torch
import torch.distributed as dist

from deepep_adapter import validate_plan
from deepep_v1_adapter import DeepEPV1PlannedAdapter, NCCLReplicaPrefetch, UPSTREAM_COMMIT, build_provenance


def diagnose(*,output,rank,no_cpu_sync,logical,physical,mapping,recv_x,recv_idx,
             recv_router,counts,handle,local_factors,contribution,out,combined,expected,
             send_head_before,send_head_expected):
    """Retain exact synthetic tensors on failure; never change the acceptance test."""
    error=(combined.float()-expected[:,None]).abs()
    mismatch=error.amax(dim=1)>0
    ids=mismatch.nonzero().flatten()[:16]
    valid=recv_idx>=0
    src=handle[3].long()
    src_in_range=(src>=0)&(src<logical.shape[0])
    local_slots=local_factors.numel()
    expected_recv=physical[src.clamp(0,logical.shape[0]-1)]
    expected_recv=torch.where(expected_recv//local_slots==rank,
                              expected_recv%local_slots,-1)
    usable=src_in_range&valid.any(dim=1)
    detail=dict(no_cpu_sync=no_cpu_sync,max_abs_error=float(error.max().item()),
        mismatched_output_tokens=int(mismatch.sum().item()),
        first_mismatch_ids=ids.cpu().tolist(),
        first_expected=expected[ids].cpu().tolist(),
        first_combined=combined[ids,:8].float().cpu().tolist(),
        output_min=float(combined.min().item()),output_max=float(combined.max().item()),
        recv_shape=list(recv_x.shape),recv_payload_all_ones=bool((recv_x[valid.any(dim=1)]==1).all()),
        recv_valid_router_is_one_over_k=bool((recv_router[valid]==1/logical.shape[1]).all()),
        recv_idx_min=int(recv_idx.min().item()),recv_idx_max=int(recv_idx.max().item()),
        recv_src_index_valid=int(src_in_range.sum().item()),
        recv_exact_expert_slots_on_valid_rows=bool((recv_idx[usable]==expected_recv[usable]).all()),
        local_factors=local_factors.float().cpu().tolist(),
        actual_recv_expert_counts=counts,
        rank_prefix=handle[0].cpu().tolist(),
        valid_counts_histogram=torch.bincount(valid.sum(dim=1),minlength=logical.shape[1]+1).cpu().tolist(),
        input_expected_min=float(expected.min().item()),input_expected_max=float(expected.max().item()),
        contribution_min=float(contribution.min().item()),contribution_max=float(contribution.max().item()))
    detail.update(send_head_before_exact=bool((send_head_before==send_head_expected).all()),
                  send_head_before=send_head_before.cpu().tolist(),
                  send_head_expected=send_head_expected.cpu().tolist(),
                  send_head_after=handle[5].cpu().tolist(),
                  first_bad_expected_heads=send_head_expected[ids].cpu().tolist(),
                  first_bad_before_heads=send_head_before[ids].cpu().tolist(),
                  first_bad_after_heads=handle[5][ids].cpu().tolist())
    if bool(mismatch.any()):
        tensor_path=output.with_name(output.stem+f'_rank{rank}_sync{int(not no_cpu_sync)}_failure_tensors.pt')
        tensor_path.parent.mkdir(parents=True,exist_ok=True)
        tensors=dict(logical=logical,physical=physical,mapping=mapping,recv_x=recv_x,
                     recv_idx=recv_idx,recv_router=recv_router,local_factors=local_factors,
                     contribution=contribution,expert_output=out,combined=combined,
                     expected=expected,send_head_before=send_head_before,
                     send_head_expected=send_head_expected,
                     **{f'handle_{i}':t for i,t in enumerate(handle) if isinstance(t,torch.Tensor)})
        torch.save({key:value.detach().cpu() for key,value in tensors.items()},tensor_path)
        detail['failure_tensor_path']=str(tensor_path)
    return detail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--weight-elements", type=int, default=4096)
    p.add_argument('--num-experts',type=int,default=352)
    p.add_argument('--extra-slots',type=int,default=8)
    p.add_argument('--no-replica-moves',action='store_true')
    p.add_argument('--no-weight-copy',action='store_true',help='routing-only control; synthesize replica markers locally')
    p.add_argument("--output", type=Path, default=Path("deepep_v1_smoke.json"))
    a = p.parse_args()
    if a.num_experts<8 or a.num_experts%2 or a.extra_slots<0:
        p.error('an even number of experts >=8 and nonnegative replica slots are required')
    if not a.no_replica_moves and a.extra_slots==0:
        p.error('use --no-replica-moves when extra slots are zero')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    weight_group = None if a.no_weight_copy else dist.new_group([0, 1], backend="nccl")
    result = {"status": "running", "token_backend": "DeepEP v1.2.1 intranode",
              "expert_backend": "not exercised; locally filled synthetic markers" if a.no_weight_copy else "torch.distributed NCCL batch P2P",
              "reference_commit": UPSTREAM_COMMIT,
              "workload": "synthetic correctness probe", "failures": [], "checks": [], "diagnostics": []}
    result['shape_and_controls']=dict(tokens=a.tokens,hidden=a.hidden,num_experts=a.num_experts,
        extra_slots=a.extra_slots,no_replica_moves=a.no_replica_moves,no_weight_copy=a.no_weight_copy)
    adapter = None
    try:
        result['build_provenance'] = build_provenance()
        E, K, B, local = a.num_experts, 8, a.extra_slots, a.num_experts//2
        adapter = DeepEPV1PlannedAdapter(dist.group.WORLD, per_source=a.tokens,
            hidden=a.hidden, topk=K, num_experts=E, extra_slots=B)
        cpu_map = torch.full((2, B), -1, dtype=torch.int32)
        if not a.no_replica_moves:
            cpu_map[0, 0], cpu_map[1, 0] = local, 0
        mapping = cpu_map.cuda()
        ids = torch.arange(local, device="cuda") + rank * local
        factors = ((ids % 7) + 1).to(torch.bfloat16)
        original = factors[:, None].expand(-1, a.weight_elements).contiguous()
        replicas = torch.full((B, a.weight_elements), -1, device="cuda", dtype=torch.bfloat16)
        prefetch = None if a.no_weight_copy else NCCLReplicaPrefetch(weight_group, original, replicas, cpu_map)
        if a.no_weight_copy:
            for slot,e in enumerate(cpu_map[rank].tolist()):
                if e>=0:replicas[slot].fill_(e%7+1)
        logical = (torch.arange(a.tokens*K, device="cuda").reshape(-1, K) % E).int()
        owner = logical // local
        if not a.no_replica_moves:
            owner = torch.where(logical == 0, 1, torch.where(logical == local, 0, owner)).int()
        validate_plan(logical, owner, mapping, E)
        x = torch.ones((a.tokens, a.hidden), dtype=torch.bfloat16, device="cuda")
        router = torch.full((a.tokens, K), 1/K, dtype=torch.float32, device="cuda")
        for no_cpu_sync in (False, True):
            torch.cuda.synchronize()
            dist.barrier()
            weights = prefetch.start() if prefetch is not None else None
            token = adapter.start_token_dispatch(x, logical, owner, mapping, router,
                                                 no_cpu_sync=no_cpu_sync)
            recv_x, recv_idx, recv_router, counts, handle = token.wait()
            if weights is not None:weights.wait()
            send_head_before=handle[5].clone()
            send_head_expected=torch.empty_like(send_head_before)
            channels=handle[1].shape[1]
            chunk=(a.tokens+channels-1)//channels
            for channel in range(channels):
                begin,end=min(channel*chunk,a.tokens),min((channel+1)*chunk,a.tokens)
                present=torch.stack([(owner[begin:end]==target).any(dim=1) for target in range(2)],dim=1)
                positions=present.int().cumsum(dim=0)-1
                send_head_expected[begin:end]=torch.where(present,positions,-1).to(send_head_expected.dtype)
            # v1 receive rows are deduplicated by token/destination rank. Sum all
            # valid local expert contributions for each row before combine.
            local_factors = torch.cat((original[:, 0], replicas[:, 0]))
            valid = recv_idx >= 0
            selected = local_factors[recv_idx.clamp_min(0)]
            contribution = torch.where(valid, selected.float() * recv_router, 0).sum(dim=1)
            out = (recv_x.float() * contribution[:, None]).bfloat16()
            # Avoid NaN propagation if capacity rows contain unspecified payload.
            out.masked_fill_(~valid.any(dim=1)[:, None], 0)
            combined, _, event = adapter.combine(out, handle)
            event.current_stream_wait()
            expected = (((logical % 7) + 1).float()/K).sum(dim=1)
            torch.cuda.synchronize()
            error = (combined.float() - expected[:, None]).abs().max().item()
            result['diagnostics'].append(diagnose(output=a.output,rank=rank,no_cpu_sync=no_cpu_sync,
                logical=logical,physical=token.physical,mapping=mapping,recv_x=recv_x,
                recv_idx=recv_idx,recv_router=recv_router,counts=counts,handle=handle,
                local_factors=local_factors,contribution=contribution,out=out,
                combined=combined,expected=expected,send_head_before=send_head_before,
                send_head_expected=send_head_expected))
            assert error == 0, error
            for slot,e in enumerate(cpu_map[rank].tolist()):
                assert bool((replicas[slot] == (e%7+1 if e>=0 else -1)).all())
            result["checks"].append(dict(no_cpu_sync=no_cpu_sync, max_abs_error=error,
                replica_value_verified=True, unused_slots_unchanged=True))
        result.update(status="passed", weight_bytes_sent=prefetch.bytes_sent if prefetch else 0,
                      weight_bytes_received=prefetch.bytes_received if prefetch else 0)
    except Exception as exc:
        result["status"] = "failed"
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        result['traceback']=traceback.format_exc()
        raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_name(a.output.stem+f"_rank{rank}.json").write_text(
            json.dumps(result, indent=2)+"\n")
        if adapter is not None:
            adapter.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
