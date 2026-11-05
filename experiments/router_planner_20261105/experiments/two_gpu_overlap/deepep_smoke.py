"""Two physical GPU correctness smoke for real DeepEP; no performance claims.

torchrun --standalone --nproc_per_node=2 experiments/two_gpu_overlap/deepep_smoke.py
Add --independent after applying/rebuilding lb-current-stream.patch.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from deepep_adapter import DeepEPPlannedAdapter, capabilities, validate_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--independent", action="store_true")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--weight-elements", type=int, default=4096)
    p.add_argument("--output", type=Path, default=Path("deepep_smoke.json"))
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    E, K, B, local = 352, 8, 8, 176
    adapter = None
    result = {"status": "running", "capabilities": capabilities(),
              "mode": "independent" if a.independent else "upstream_same_stream",
              "workload": "synthetic correctness only", "failures": []}
    try:
        adapter = DeepEPPlannedAdapter(
            dist.group.WORLD, per_source=a.tokens, hidden=a.hidden, topk=K,
            num_experts=E, extra_slots=B, weight_shapes=[(a.weight_elements,)],
            expert_alignment=1, independent_weight_stream=a.independent)
        mapping = torch.full((2, B), -1, dtype=torch.int32, device="cuda")
        mapping[0, 0], mapping[1, 0] = local, 0
        # Positive exactly representable per-expert multiplier identifies copies.
        ids = torch.arange(local, device="cuda") + rank * local
        factors = ((ids % 7) + 1).to(torch.bfloat16)
        adapter.expert_weights[0].copy_(factors[:, None].expand(-1, a.weight_elements))
        adapter.replica_weights[0].fill_(-1)
        logical = (torch.arange(a.tokens * K, device="cuda").reshape(-1, K) % E).int()
        owner = logical // local
        owner = torch.where(logical == 0, 1, torch.where(logical == local, 0, owner)).int()
        validate_plan(logical, owner, mapping, E)
        x = torch.ones((a.tokens, a.hidden), dtype=torch.bfloat16, device="cuda")
        router = torch.full((a.tokens, K), 1.0 / K, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        dist.barrier()
        weights = adapter.start_expert_prefetch(mapping)
        tokens = adapter.start_token_dispatch(x, logical, owner, mapping, router,
                                               do_cpu_sync=True)
        recv_x, _, recv_router, handle = adapter.wait_for_expert_compute(weights, tokens)
        torch.cuda.synchronize()
        expected_replica = float((int(mapping[rank, 0]) % 7) + 1)
        assert bool((adapter.replica_weights[0][0] == expected_replica).all())
        assert bool((adapter.replica_weights[0][1:] == -1).all())
        # Toy expert uses the actual received replica value. No GEMM is claimed.
        out = torch.zeros_like(recv_x)
        counts = handle.num_recv_tokens_per_expert_list
        assert len(counts) == local + B
        start = 0
        for slot, count in enumerate(counts):
            if count:
                multiplier = (adapter.expert_weights[0][slot, 0] if slot < local
                              else adapter.replica_weights[0][slot-local, 0])
                out[start:start+count] = (recv_x[start:start+count].float() *
                    recv_router[start:start+count, None] * multiplier.float()).bfloat16()
            start += count
        combined, _ = adapter.combine(out, handle).wait()
        expected = (((logical % 7) + 1).float() / K).sum(dim=1)
        torch.cuda.synchronize()
        error = (combined.float() - expected[:, None]).abs().max().item()
        assert error == 0, error
        result.update(status="passed", max_abs_error=error,
                      replica_value_verified=True, unused_slots_unchanged=True,
                      dispatch_combine_verified=True)
    except Exception as exc:
        result["status"] = "failed"
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_name(a.output.stem + f"_rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n")
        if adapter is not None:
            adapter.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
