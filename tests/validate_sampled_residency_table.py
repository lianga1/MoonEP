"""Single-GPU exact regression for the cached-group greedy selector.

Run from the built source tree, without torchrun:
  python -m tests.validate_sampled_residency_table \
    --fixture-root /path/to/kernel_inputs --output /path/to/result.json

The default cases are the original eight B8 trace layers, a B44 version of
layer 0, and E520/K16/D512/G3/B8 (whose arrays alone exceed 48 KiB, forcing
the generic fallback). Optional cases are small=E520/K3/D37/G3/B8,
highg=E128/K1/D128/G8/B8 and k16=E520/K16/D128/G1/B8. Each synthetic case
replays the same Graph with two different router inputs.
All source packets are gathered and summed on this one GPU; there is no
distributed-performance claim. CPU oracles run once per input, outside Graph.
No timings from this correctness test are benchmark results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from moonep.sampled_residency import Residency
from tests.sampled_residency_reference import reference


SEED = 20261026


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic(experts, topk, samples, groups, seed):
    """Repeated home/striped/skewed groups with legal home-only seed owners."""
    rng = np.random.default_rng(seed)
    per_source, home = 64, experts // 8
    prototypes = []
    for i in range(48):
        homes = np.full(topk, i % 8, np.int32)
        if (i // 8) % 3 == 1:
            homes = (i + np.arange(topk)) % 8
        elif (i // 8) % 3 == 2:
            homes[-max(1, topk // 4):] = (i + 1) % 8
        row = np.empty(topk, np.int32)
        for h in np.unique(homes):
            pos = np.flatnonzero(homes == h)
            row[pos] = h * home + rng.choice(home, len(pos), replace=False)
        prototypes.append(row)
    prototypes = np.asarray(prototypes, np.int32)
    routes = prototypes[rng.integers(len(prototypes), size=(8, per_source))].copy()
    ids = np.sort(np.random.default_rng(SEED).choice(8 * per_source, samples, replace=False)).astype(np.int64)
    routes.reshape(-1, topk)[ids[:min(16, samples)]] = prototypes[8]
    return dict(routes=routes, owner=(routes // home).astype(np.int32),
                copies=np.full((8, 8), -1, np.int32), ids=ids,
                experts=experts, groups=groups)


def trace_input(root, layer, slots=8):
    path = root / f"layer_{layer:02d}.npz"
    with np.load(path, allow_pickle=False) as z:
        routes, owner = z["topk"].copy(), z["seed"].copy()
        native, ids, support = z["native_experts_to_copy"].copy(), z["sample_ids"].copy(), z["support"].copy()
    if routes.shape != (8, 16768, 8) or owner.shape != routes.shape or native.shape != (8, 8):
        raise ValueError(f"unexpected original trace shape in {path}")
    copies = np.full((8, slots), -1, np.int32)
    copies[:, :8] = native
    return dict(routes=routes, owner=owner, copies=copies, ids=ids,
                experts=352, groups=2, saved_support=support if slots == 8 else None,
                fixture_sha256=sha(path))


class Runner:
    def __init__(self, data):
        self.device = torch.device("cuda", 0)
        _, per_source, topk = data["routes"].shape
        self.planner = Residency(dict(R=8, E=data["experts"], K=topk, S=per_source,
            B=data["copies"].shape[1], rank=0, device=self.device), SEED,
            sample_size=len(data["ids"]), groups_per_class=data["groups"], component_only=True)
        np.testing.assert_array_equal(self.planner.ids.cpu().numpy(), data["ids"])
        self.routes = torch.empty(data["routes"].shape, dtype=torch.int32, device=self.device)
        self.owner = torch.empty_like(self.routes)
        self.copies = torch.empty(data["copies"].shape, dtype=torch.int32, device=self.device)
        self.route_views, self.owner_views = list(self.routes.unbind(0)), list(self.owner.unbind(0))
        self.packet = torch.empty_like(self.planner.packet)
        self.graph = None

    def load(self, data):
        np.testing.assert_array_equal(self.planner.ids.cpu().numpy(), data["ids"])
        for name in ("routes", "owner", "copies"):
            getattr(self, name).copy_(torch.as_tensor(data[name], device=self.device))

    def execute(self):
        self.packet.zero_()
        for source in range(8):
            self.planner.gather_local(self.route_views[source], self.owner_views[source],
                                      self.copies, source_rank=source)
            self.packet.add_(self.planner.packet)
        self.planner.build_from_reduced_samples(self.copies, self.packet)

    def poison(self):
        for name in ("samples", "copy_seen", "extra", "classes", "groups", "sig0", "sig1",
                     "masks", "frequencies", "tops", "first", "records", "sizes", "package_offsets",
                     "packages", "package_count", "experts_to_copy", "history", "history_count", "status"):
            getattr(self.planner, name).fill_(-123)
        self.planner.support.zero_()
        self.packet.fill_(-123)

    def close(self):
        if self.graph is not None:
            torch.cuda.synchronize()
            self.graph.reset()
            self.graph = None


def assert_exact(planner, expected, label):
    assert int(planner.status.item()) == 0, (label, "device status")
    for name in ("support", "extra", "classes", "groups", "frequencies", "tops", "first",
                 "records", "sizes", "package_offsets", "sig0", "sig1", "masks"):
        actual, wanted = getattr(planner, name).cpu().numpy(), expected[name]
        if wanted.dtype == np.uint64:
            actual = actual.view(np.uint64)
        np.testing.assert_array_equal(actual, wanted, err_msg=f"{label}: {name}")
    package_count, history_count = int(planner.package_count.item()), int(planner.history_count.item())
    assert package_count == len(expected["packages"]), (label, "package count")
    assert history_count == len(expected["history_rows"]), (label, "history count")
    np.testing.assert_array_equal(planner.packages[:package_count].cpu().numpy(), expected["packages"],
                                  err_msg=f"{label}: package order")
    # Compare every defined raw field, including original package index, cost
    # and all expert-addition bit words. Signed CUDA words are compared by bits.
    history = planner.history[:history_count].cpu().numpy().view(np.uint64)
    wanted_history = np.asarray(expected["history_rows"], np.uint64).reshape(history_count, planner.history_width)
    np.testing.assert_array_equal(history, wanted_history, err_msg=f"{label}: complete raw history")
    assert planner.accepted_packages() == expected["history"], (label, "decoded history")
    copies = np.full((8, planner.slots), -1, np.int32)
    home = np.arange(planner.experts) // (planner.experts // 8)
    for rank in range(8):
        ids = np.flatnonzero(expected["support"][rank] & (home != rank))
        copies[rank, :len(ids)] = ids
    np.testing.assert_array_equal(planner.experts_to_copy.cpu().numpy(), copies,
                                  err_msg=f"{label}: complete padded copy list")
    return dict(packages=package_count, accepted=history_count,
                raw_history_sha256=hashlib.sha256(history.tobytes()).hexdigest(),
                complete_raw_history_exact=True, all_intermediates_exact=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture-root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", default="trace,b44,fallback")
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--replays", type=int, default=2)
    args = ap.parse_args()
    cases, layers = args.cases.split(","), [int(i) for i in args.layers.split(",")]
    if not cases or set(cases) - {"trace", "b44", "fallback", "small", "highg", "k16"} or args.replays < 1:
        ap.error("cases must be trace,b44,fallback,small,highg,k16; replays must be positive")
    if set(cases) & {"trace", "b44"} and args.fixture_root is None:
        ap.error("trace/b44 require --fixture-root")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        ap.error("ordinary single-process execution only; do not launch with torchrun")
    report = dict(status="running", scope="single GPU, all eight logical source packets; selector correctness only",
                  cases=[], failures=[], timings_are_benchmarks=False)
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    runner = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("one CUDA device is required")
        torch.cuda.set_device(0)
        report.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                      test_sha256=sha(__file__), oracle_sha256=sha(Path(__file__).with_name("sampled_residency_reference.py")))
        cohorts = []
        if "trace" in cases:
            cohorts.append(("trace", [(f"layer_{i:02d}", lambda i=i: trace_input(args.fixture_root, i)) for i in layers]))
        if "b44" in cases:
            cohorts.append(("b44", [("layer_00_B44", lambda: trace_input(args.fixture_root, 0, 44))]))
        for case, shape in (("fallback", (520, 16, 512, 3)), ("small", (520, 3, 37, 3)),
                            ("highg", (128, 1, 128, 8)), ("k16", (520, 16, 128, 1))):
            if case in cases:
                cohorts.append((case, [(f"{case}_seed{seed}", lambda shape=shape, seed=seed: synthetic(*shape, seed))
                                      for seed in (932071, 932072)]))
        for cohort, inputs in cohorts:
            for name, make_input in inputs:
                data = make_input()
                row = dict(name=name, cohort=cohort, passes=[])
                report["cases"].append(row)
                save()
                started = time.monotonic()
                expected = reference(data["routes"], data["owner"], data["copies"], data["ids"],
                                     experts=data["experts"], groups_per_class=data["groups"])
                row["cpu_oracle_seconds"] = time.monotonic() - started
                assert len(expected["history_rows"]) > 0, "fixture must exercise greedy selection"
                if cohort == "b44":
                    assert len(expected["history_rows"]) > 64, "B44 case must exceed the former B8 history limit"
                if data.get("saved_support") is not None:
                    np.testing.assert_array_equal(expected["support"], data["saved_support"])
                if runner is None:
                    runner = Runner(data)
                    report["production_sources"] = runner.planner.source_manifest()
                runner.load(data)
                planner = runner.planner
                row.update(E=planner.experts, K=planner.topk, D=planner.sample_size,
                           G=planner.groups_per_class, B=planner.slots,
                           fixture_sha256=data.get("fixture_sha256"))
                arrays = 12 * planner.buckets * 2 * planner.groups_per_class + 8 * planner.buckets + 64 * planner.words + 48
                row["shared_array_bytes_excluding_CUB_alignment"] = arrays
                row["expected_path"] = "generic (>48KiB arrays alone)" if arrays > 49152 else "table (compile must confirm CUB fits)"
                runner.execute()
                torch.cuda.synchronize()
                row["passes"].append(dict(mode="eager", **assert_exact(planner, expected, name)))
                if runner.graph is None:
                    runner.execute()
                    torch.cuda.synchronize()
                    runner.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(runner.graph):
                        runner.execute()
                for replay in range(args.replays):
                    runner.poison()
                    runner.graph.replay()
                    torch.cuda.synchronize()
                    row["passes"].append(dict(mode=f"graph_poisoned_{replay}", **assert_exact(planner, expected, name)))
                # Hash collisions must not merge different exact expert groups.
                if name == inputs[0][0]:
                    planner.sig0.zero_(); planner.sig1.zero_(); planner.frequencies.fill_(-1)
                    planner.ext.count_frequencies(planner.classes, planner.groups, planner.sig0,
                                                  planner.sig1, planner.frequencies)
                    np.testing.assert_array_equal(planner.frequencies.cpu().numpy(), expected["frequencies"])
                    row["forced_hash_collision_exact"] = True
                    # A malformed native list must skip greedy metadata/selection,
                    # report rejection, and recover on the same captured Graph.
                    runner.copies[0, 0] = planner.experts
                    runner.graph.replay(); torch.cuda.synchronize()
                    rejected = int(planner.status.item())
                    assert rejected & 1 and int(planner.history_count.item()) == 0
                    row["invalid_copy_status"] = rejected
                    runner.load(data); runner.poison(); runner.graph.replay(); torch.cuda.synchronize()
                    row["passes"].append(dict(mode="recovered_after_invalid_copy", **assert_exact(planner, expected, name)))
                save()
                print(json.dumps(dict(name=name, packages=len(expected["packages"]), accepted=len(expected["history_rows"]),
                                      cpu_oracle_seconds=row["cpu_oracle_seconds"], passed=True)), flush=True)
            runner.close()
            runner = None
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["failures"].append(dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        save()
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
