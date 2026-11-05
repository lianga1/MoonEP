"""Two physical GPUs, eight ORIGINAL logical sources: component correctness.

Run from the built MoonEP checkout:
  torchrun --standalone --nproc_per_node=2 tests/sampled_components.py \
    --fixture-root /path/to/eight_rank_estimate/kernel_inputs \
    --output /path/to/unique_attempt.json

Only the installed/built moonep package, NumPy, Torch and explicit fixtures are
required. No workspace-only reference module is imported. B=8 oracle fixtures
contain original CPU support and all five owner prefixes, including the seed.
The optional fixture manifest supplies the original full residency history.

Each physical process holds four source shards. Samples use a REAL two-rank
SUM; every exchange round constructs live local counts and uses a REAL two-rank
all-gather before quotas/commits. No oracle counts enter the runtime. The same
captured CUDA Graph is replayed with changed layer inputs and poisoned scratch.
This is NOT native eight-physical-rank execution or an end-to-end benchmark.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixture(root, layer):
    path = root / f"layer_{layer:02d}.npz"
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    required = {"topk", "seed", "support", "owner_prefixes", "stats", "native_experts_to_copy", "sample_ids"}
    if not required <= result.keys():
        raise ValueError(f"{path}: missing fields {required - result.keys()}")
    if result["topk"].shape != (8, 16768, 8) or result["owner_prefixes"].shape != (5, 8, 16768, 8):
        raise ValueError(f"{path}: fixture is not the original N=134144/R8/K8 input")
    if result["native_experts_to_copy"].shape != (8, 8) or result["support"].shape != (8, 352):
        raise ValueError(f"{path}: fixture must use B=8 and E=352")
    if result["stats"].shape != (4, 4) or result["sample_ids"].shape != (256,):
        raise ValueError(f"{path}: invalid four-round stats or global sample IDs")
    if not np.array_equal(result["owner_prefixes"][0], result["seed"]):
        raise ValueError(f"{path}: first oracle prefix is not the native seed")
    return result, dict(path=str(path), sha256=sha256(path))


def remote_count(owner, sources):
    sorted_owner = np.sort(owner, axis=-1)
    first = np.concatenate((np.ones((*owner.shape[:-1], 1), dtype=bool),
                            sorted_owner[..., 1:] != sorted_owner[..., :-1]), axis=-1)
    source_ids = np.asarray(sources)[:, None, None]
    return int(np.count_nonzero(first & (sorted_owner != source_ids)))


class Components:
    def __init__(self, rank, device, Residency, SampledExchange, buckets):
        self.rank, self.device = rank, device
        self.sources = list(range(4 * rank, 4 * rank + 4))
        self.routes = torch.empty((4, 16768, 8), dtype=torch.int32, device=device)
        self.seed = torch.empty_like(self.routes)
        self.native_copies = torch.empty((8, 8), dtype=torch.int32, device=device)
        ctx = dict(R=8, E=352, K=8, S=16768, B=8, rank=self.sources[0], device=device, group=None)
        self.residency = Residency(ctx, sample_seed=20261026, component_only=True)
        self.local_routes = list(self.routes.unbind(0))
        self.local_seed = list(self.seed.unbind(0))
        self.exchanges = [SampledExchange(self.local_routes[i], self.residency.support,
            source=source, rounds=4) for i, source in enumerate(self.sources)]
        self.sample_sum = torch.empty_like(self.residency.packet)
        self.local_counts = torch.empty((4, buckets), dtype=torch.int32, device=device)
        self.global_counts = torch.empty((8, buckets), dtype=torch.int32, device=device)
        self.count_history = torch.empty((4, 8, buckets), dtype=torch.int32, device=device)
        self.owner_prefixes = torch.empty((4, 4, 16768, 8), dtype=torch.int32, device=device)

    def inputs(self, data):
        # Fixture input copies happen outside the captured runtime. Expected
        # support/owners/stats are never used to make a runtime decision.
        self.routes.copy_(torch.as_tensor(np.ascontiguousarray(data["topk"][self.sources]), device=self.device))
        self.seed.copy_(torch.as_tensor(np.ascontiguousarray(data["seed"][self.sources]), device=self.device))
        self.native_copies.copy_(torch.as_tensor(data["native_experts_to_copy"], device=self.device))

    def poison_scratch(self):
        self.sample_sum.fill_(-301)
        self.residency.packet.fill_(-302)
        self.residency.support.zero_()
        self.residency.status.fill_(31)
        self.residency.frequencies.fill_(-303)
        self.residency.history_count.fill_(-304)
        self.local_counts.fill_(-305)
        self.global_counts.fill_(-306)
        self.count_history.fill_(-307)
        self.owner_prefixes.fill_(-308)
        for exchange in self.exchanges:
            exchange.owner.fill_(-309)
            exchange.old_masks.fill_(-310)
            exchange.exec_masks.fill_(-311)
            exchange.counts.fill_(-312)
            exchange.stats.fill_(-313)

    def run(self):
        self.sample_sum.zero_()
        for i, source in enumerate(self.sources):
            self.residency.gather_local(self.local_routes[i], self.local_seed[i], self.native_copies,
                                         source_rank=source)
            self.sample_sum.add_(self.residency.packet)
        dist.all_reduce(self.sample_sum, op=dist.ReduceOp.SUM)
        self.residency.build_from_reduced_samples(self.native_copies, self.sample_sum)
        for i, exchange in enumerate(self.exchanges):
            exchange.prepare_support()
            exchange.reset(self.local_seed[i])
        for step in range(4):
            for i, exchange in enumerate(self.exchanges):
                exchange.propose(step)
                self.local_counts[i].copy_(exchange.counts)
            # Concatenation by physical rank preserves original source order:
            # physical 0 owns sources 0..3; physical 1 owns sources 4..7.
            dist.all_gather_into_tensor(self.global_counts, self.local_counts)
            self.count_history[step].copy_(self.global_counts)
            for i, exchange in enumerate(self.exchanges):
                exchange.commit(step, self.global_counts)
                self.owner_prefixes[step, i].copy_(exchange.owner)

    def audit(self, data, expected_history):
        errors = []
        status = int(self.residency.status.item())
        if status != 0:
            errors.append(f"residency status={status}")
        actual_support = self.residency.support.cpu().numpy()
        if not np.array_equal(actual_support, data["support"]):
            errors.append(f"support mismatch entries={np.count_nonzero(actual_support != data['support'])}")
        ids = data["sample_ids"]
        expected_samples = np.concatenate((data["topk"].reshape(-1, 8)[ids], data["seed"].reshape(-1, 8)[ids]), axis=1)
        if not np.array_equal(self.residency.samples.cpu().numpy(), expected_samples):
            errors.append("runtime gathered routes/owners differ from current layer samples")
        actual = self.owner_prefixes.cpu().numpy()
        expected = data["owner_prefixes"][1:, self.sources]
        for step in range(4):
            if not np.array_equal(actual[step], expected[step]):
                errors.append(f"round {step} owner mismatch entries={np.count_nonzero(actual[step] != expected[step])}")
        # stats are global summaries repeated at every source; summing eight
        # source copies would multiply the actual global statistics by eight.
        all_stats = [exchange.stats.cpu().numpy() for exchange in self.exchanges]
        for source, stats in zip(self.sources, all_stats):
            if not np.array_equal(stats, data["stats"]):
                errors.append(f"source {source}: global round stats mismatch")
        if expected_history is not None and self.residency.accepted_packages() != expected_history:
            errors.append("residency accepted-package history differs original CPU oracle")
        copies = self.residency.experts_to_copy.cpu().numpy()
        for rank in range(8):
            expected_ids = np.flatnonzero(data["support"][rank] & (np.arange(352) // 44 != rank))
            if not np.array_equal(copies[rank][copies[rank] >= 0], expected_ids):
                errors.append(f"rank {rank}: output copy list mismatch")
        counts = self.count_history.cpu().numpy()
        if np.any(counts < 0):
            errors.append("live gathered counts retain poisoned negative values")
        before = remote_count(data["seed"][self.sources], self.sources)
        after = remote_count(actual[-1], self.sources)
        return dict(physical_rank=self.rank, logical_sources=self.sources, errors=errors,
            status=status, remote_before_local=before, remote_after_local=after,
            runtime_count_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
            global_stats_once=all_stats[0].tolist(),
            global_stats_sum_over_rounds_once=all_stats[0].sum(axis=0).tolist(),
            accepted_history_length=int(self.residency.history_count.item()),
            history_checked=expected_history is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    layers = list(map(int, args.layers.split(",")))
    if not layers or len(set(layers)) != len(layers) or any(i not in range(8) for i in layers):
        raise ValueError("layers must be unique original layer IDs 0..7")
    if args.repeats < 2 or args.warmup < 1:
        raise ValueError("at least two reset/replays and one warm-up are required")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    graph = None
    can_write_output = False
    record = dict(status="running", scope="two physical GPUs; eight original logical sources; component correctness only",
        not_eight_physical_rank_runtime=True, not_end_to_end_or_performance_measurement=True,
        protocol=dict(N=134144, R=8, E=352, K=8, B=8, d=256, g=2, L=4, seed=20261026,
            layers=layers, repeats=args.repeats, warmup=args.warmup,
            sample_collective="real world2 SUM of four original-source packets per physical rank",
            exchange_collective="real world2 all-gather of four live original-source count rows per rank, every round",
            preloaded_counts_used=False, graph_reused_across_changed_layer_inputs=True),
        cases=[], failures=[])
    def save():
        if rank == 0 and can_write_output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    def gather_objects(value):
        received = [None, None]
        dist.all_gather_object(received, value)
        return received
    def collective_errors(errors, phase):
        gathered = gather_objects(dict(rank=rank, errors=errors))
        bad = [row for row in gathered if row["errors"]]
        if bad:
            record["failures"].append(dict(phase=phase, ranks=bad))
            save()
            raise AssertionError(f"{phase}: {bad}")
    try:
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
        if dist.get_world_size() != 2:
            raise ValueError("this CLI must run with exactly two physical processes")
        collective_errors(["output exists; use a fresh attempt path"] if args.output.exists() else [], "preflight_output")
        can_write_output = True
        residency_module = importlib.import_module("moonep.sampled_residency")
        exchange_module = importlib.import_module("moonep.sampled_exchange")
        record["environment"] = dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
            nccl=list(torch.cuda.nccl.version()), devices=gather_objects(torch.cuda.get_device_name(device)))
        sources = [Path(__file__), Path(residency_module.__file__), Path(exchange_module.__file__)]
        sources.extend(Path(residency_module.__file__).parent / "csrc" / name
                       for name in ("sampled_residency.cu", "sampled_exchange.cu"))
        record["source_sha256"] = {str(path): sha256(path) for path in sources}
        manifest_path = args.fixture_root / "manifest.json"
        histories = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            histories = {row["layer"]: row["residency_diagnostics"]["accepted_packages"] for row in manifest["layers"]}
            record["fixture_manifest_sha256"] = sha256(manifest_path)
        save()
        initial, _ = load_fixture(args.fixture_root, layers[0])
        pipeline = Components(rank, device, residency_module.Residency, exchange_module.SampledExchange, exchange_module.BUCKETS)
        collective_errors([] if np.array_equal(pipeline.residency.ids.cpu().numpy(), initial["sample_ids"])
                          else ["constructor sample IDs differ from fixture"], "sample_ids")
        pipeline.inputs(initial)
        for _ in range(args.warmup):
            pipeline.run()
        torch.cuda.synchronize()
        first_audit = pipeline.audit(initial, histories.get(layers[0]))
        collective_errors(first_audit["errors"], "eager_warmup")
        record["eager_warmup"] = gather_objects(first_audit)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline.run()
        # The Graph retains these same buffers/objects for every layer below.
        for layer in layers:
            data, provenance = load_fixture(args.fixture_root, layer)
            case = dict(layer=layer, fixture=provenance, passes=[])
            record["cases"].append(case)
            pipeline.inputs(data)
            collective_errors([] if np.array_equal(data["sample_ids"], initial["sample_ids"])
                              else ["sample IDs changed; this fixed-index Graph protocol requires the same seed"], f"layer{layer}_ids")
            for repeat in range(args.repeats):
                pipeline.poison_scratch()
                graph.replay()
                torch.cuda.synchronize()
                audit = pipeline.audit(data, histories.get(layer))
                ranks = gather_objects(audit)
                errors = [f"rank {row['physical_rank']}: {error}" for row in ranks for error in row["errors"]]
                total_before = sum(row["remote_before_local"] for row in ranks)
                total_after = sum(row["remote_after_local"] for row in ranks)
                expected_before = remote_count(data["seed"], range(8))
                expected_after = remote_count(data["owner_prefixes"][-1], range(8))
                if (total_before, total_after) != (expected_before, expected_after):
                    errors.append("actual remote communication total differs original eight-source fixture")
                if ranks[0]["runtime_count_sha256"] != ranks[1]["runtime_count_sha256"]:
                    errors.append("physical ranks disagree on live global counts")
                predicted_saved = int(np.sum(data["stats"][:, 3]))
                if total_before - total_after != predicted_saved:
                    errors.append("global sum of predicted saved remote destinations differs actual count reduction")
                attempt = dict(repeat=repeat, passed=not errors, ranks=ranks,
                    remote_before=total_before, remote_after=total_after,
                    actual_saved=total_before-total_after, stats_saved_sum_once=predicted_saved,
                    oracle_counts_never_used_at_runtime=True, graph_recomputed_after_poison=True, errors=errors)
                case["passes"].append(attempt)
                save()
                collective_errors(errors, f"layer{layer}_graph_repeat{repeat}")
            if rank == 0:
                print(json.dumps(dict(layer=layer, passed=True, repeats=args.repeats,
                    remote_before=total_before, remote_after=total_after)), flush=True)
            del data
        record["status"] = "complete"
        record["all_eight_layers_checked"] = sorted(layers) == list(range(8))
        record["all_original_source_round_owners_exact"] = True
        record["original_support_and_copies_exact"] = True
        record["graph_dynamic_layers_checked"] = len(layers) > 1
        save()
    except BaseException as exc:
        failure = dict(rank=rank, error=repr(exc), traceback=traceback.format_exc())
        record["status"] = "failed"
        record["failures"].append(failure)
        # A rank-local sidecar preserves evidence even if its peer is stuck in
        # a failed collective and cannot participate in another object gather.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sidecar = args.output.with_name(args.output.stem + f".rank{rank}.failure.{os.getpid()}.json")
        sidecar.write_text(json.dumps(failure, indent=2) + "\n")
        save()
        raise
    finally:
        # Drop Graph-owned NCCL references before process-group destruction.
        if graph is not None:
            graph.reset()
            graph = None
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
