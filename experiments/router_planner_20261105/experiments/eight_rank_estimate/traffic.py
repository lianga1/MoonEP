"""Recover original eight-rank traffic, never fold routing to physical devices.

Default: rerun native rt0/rt2 and sampled residency on CPU, compare them exactly
with archived arrays, then independently audit/count the archived L=4 exchange.
--rerun-exchange also reruns the original Numba exchange implementation.
All counts are semantic traffic counts, not measured eight-GPU timings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R, E, K, S, B, D, SEED, L = 8, 352, 8, 16768, 8, 256, 20261026, 4
METHODS = ("native_rt0", "native_rt2", "sample256_g2_bundle")
EXPECTED_REMOTE = dict(zip(METHODS, (5054165, 4771576, 3791665)))
EXPECTED_COPIES = dict(zip(METHODS, (71, 203, 507)))


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_array(array):
    """Content hash of C-order bytes; shape/dtype are stated alongside it."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def provenance(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest_file(path)}


def support_from_copies(copies, home):
    support = home[None, :] == np.arange(R)[:, None]
    for rank in range(R):
        chosen = copies[rank][copies[rank] >= 0]
        assert len(chosen) == len(np.unique(chosen))
        assert np.all(home[chosen] != rank)
        support[rank, chosen] = True
    return support


def audit_and_count(routes, owner, support, home):
    assert routes.shape == owner.shape == (R, S, K)
    assert owner.dtype.kind in "iu" and np.all((owner >= 0) & (owner < R))
    assert support.shape == (R, E)
    assert np.all(support[owner, routes]), "unsupported execution pair"
    assert np.all(support[home, np.arange(E)]), "missing home expert"
    # Every unchanged (source, token, top-k position) has exactly one owner.
    execution = np.bincount(owner.ravel().astype(np.int64) * E + routes.ravel(),
                            minlength=R * E).reshape(R, E)
    assert np.array_equal(execution.sum(1), np.full(R, S * K))
    assert np.array_equal(execution.sum(0), np.bincount(routes.ravel(), minlength=E))
    external = home[None, :] != np.arange(R)[:, None]
    reserved = support & external
    active = (execution > 0) & external
    assert np.all(active <= reserved)
    assert np.all(reserved.sum(1) <= B)
    token_all = np.zeros((R, R), dtype=np.int64)
    for source in range(R):
        for dest in range(R):
            token_all[source, dest] = np.count_nonzero(np.any(owner[source] == dest, axis=1))
    dispatch = token_all.copy()
    np.fill_diagonal(dispatch, 0)
    # Rows are original expert homes; columns are replica/execution ranks.
    expert = np.zeros((R, R), dtype=np.int64)
    active_expert = np.zeros_like(expert)
    replicas = []
    for dest in range(R):
        ids = np.flatnonzero(reserved[dest])
        replicas.append(ids.tolist())
        np.add.at(expert[:, dest], home[ids], 1)
        np.add.at(active_expert[:, dest], home[np.flatnonzero(active[dest])], 1)
    by_rank = []
    for rank in range(R):
        tx, rx = int(dispatch[rank].sum()), int(dispatch[:, rank].sum())
        by_rank.append(dict(rank=rank, dispatch_tx_tokens=tx, dispatch_rx_tokens=rx,
            combine_tx_tokens=rx, combine_rx_tokens=tx,
            dispatch_plus_combine_tx_tokens=tx + rx,
            dispatch_plus_combine_rx_tokens=tx + rx,
            expert_tx_copies=int(expert[rank].sum()),
            expert_rx_copies=int(expert[:, rank].sum()),
            active_extra_experts=int(active[rank].sum()),
            reserved_extra_experts=int(reserved[rank].sum()),
            execution_pairs=int(execution[rank].sum())))
    return dict(
        token_source_to_destination_including_local=token_all.tolist(),
        token_remote_source_to_destination=dispatch.tolist(),
        combine_remote_source_to_destination=dispatch.T.tolist(),
        expert_home_to_replica=expert.tolist(),
        active_expert_home_to_replica=active_expert.tolist(),
        selected_extra_expert_ids_by_rank=replicas,
        per_rank=by_rank,
        totals=dict(remote_token_destinations=int(dispatch.sum()),
            all_token_destinations=int(token_all.sum()),
            remote_execution_pairs=int(np.count_nonzero(owner != np.arange(R)[:, None, None])),
            expert_copies_per_full_prefetch=int(expert.sum()),
            active_extra_experts=int(active.sum()),
            reserved_extra_experts=int(reserved.sum()), execution_pairs=int(owner.size)),
        audit=dict(owner_range=True, exactly_one_owner_per_original_pair=True,
            expert_execution_conservation=True, perfectly_balanced=True,
            support_legal=True, extra_slots_within_B=True,
            all_reserved_copies_active=bool(np.array_equal(active, reserved))),
    )


def load_reference_functions(rerun_exchange):
    sys.path.insert(0, str(ROOT / "analysis"))
    from moonep_cpu import plan
    have_numba = importlib.util.find_spec("numba") is not None
    if rerun_exchange and not have_numba:
        raise RuntimeError("--rerun-exchange requires Numba; default audit needs only NumPy")
    # Reciprocal selection is ordinary Python. Its optional njit vote helpers
    # are never called. Keep the original source intact when Numba is absent.
    if not have_numba:
        stub = types.ModuleType("numba")
        stub.njit = lambda *a, **kw: a[0] if a and callable(a[0]) else lambda f: f
        sys.modules["numba"] = stub
    from vendor.residency import select_support
    optimize = None
    if rerun_exchange:
        from vendor.bucket_optimizer import optimize
    return plan, select_support, optimize, have_numba


def aggregate(rows, method):
    records = [row["methods"][method] for row in rows]
    matrix_keys = ("token_source_to_destination_including_local", "token_remote_source_to_destination",
                   "combine_remote_source_to_destination", "expert_home_to_replica",
                   "active_expert_home_to_replica")
    out = {key: np.sum([r[key] for r in records], axis=0).tolist() for key in matrix_keys}
    out["totals"] = {key: sum(r["totals"][key] for r in records) for key in records[0]["totals"]}
    out["per_rank"] = [dict(rank=rank, **{key: sum(r["per_rank"][rank][key] for r in records)
        for key in records[0]["per_rank"][rank] if key != "rank"}) for rank in range(R)]
    out["reduction_pct_vs_rt0"] = 100 * (1 - out["totals"]["remote_token_destinations"] / EXPECTED_REMOTE["native_rt0"])
    assert out["totals"]["remote_token_destinations"] == EXPECTED_REMOTE[method]
    assert out["totals"]["expert_copies_per_full_prefetch"] == EXPECTED_COPIES[method]
    assert out["totals"]["active_extra_experts"] == EXPECTED_COPIES[method]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eight_rank_estimate/traffic.json")
    parser.add_argument("--rerun-exchange", action="store_true")
    args = parser.parse_args()
    plan, select_support, optimize, have_numba = load_reference_functions(args.rerun_exchange)
    home = np.arange(E, dtype=np.int32) // (E // R)
    ids = np.sort(np.random.default_rng(SEED).choice(R * S, D, replace=False))
    evidence_path = ROOT / "outputs/planner_explainer/sampling_evidence.json"
    sampling_evidence = json.loads(evidence_path.read_text())
    original_ids = next(row["ids"] for row in sampling_evidence["seeds"] if row["seed"] == SEED)
    assert ids.tolist() == original_ids
    archive_results_path = ROOT / "outputs/rt2_b8_comparison/results.json"
    archive_results = json.loads(archive_results_path.read_text())
    rows = []
    for layer in range(8):
        trace_path = ROOT / f"outputs/trace_audit/layer_{layer:02d}.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            routes = trace["topk"]
            assert np.array_equal(trace["home"], home)
            assert not bool(trace["weights_present"])
        old_row = archive_results["rows"][layer]
        assert digest_file(trace_path) == old_row["input_sha256"]
        row = dict(layer=layer, input=provenance(trace_path), shape=list(routes.shape), methods={})
        fresh = {}
        for method, rt in (("native_rt0", 0), ("native_rt2", 2)):
            path = ROOT / f"outputs/rt2_b8_comparison/layer{layer:02d}_{method}.npz"
            replay = plan(routes, E=E, B=B, replica_target=rt)
            fresh[method] = replay
            with np.load(path, allow_pickle=False) as archived:
                for name in ("owner", "alloc", "physical_topk", "experts_to_copy"):
                    assert np.array_equal(getattr(replay, name), archived[name]), (layer, method, name)
            assert np.all(replay.physical_topk >= 0)
            support = support_from_copies(replay.experts_to_copy, home)
            record = audit_and_count(routes, replay.owner, support, home)
            record.update(archived_plan=provenance(path), replay_kind="fresh CPU native replay; all four plan arrays equal archive",
                          original_experts_to_copy=replay.experts_to_copy.tolist())
            row["methods"][method] = record
        path = ROOT / f"outputs/replay/layer{layer:02d}_b8_small256_global_4.npz"
        with np.load(path, allow_pickle=False) as archived:
            owner, support = archived["owner"], archived["support"]
        rebuilt, diagnostics = select_support(routes, fresh["native_rt0"].owner, home,
            slots=B, method="reciprocal", sample_tokens=D, groups_per_direction=2, seed=SEED)
        assert np.array_equal(rebuilt, support), (layer, "sample support")
        if optimize is not None:
            rerun, exchange_details = optimize(routes, fresh["native_rt0"].owner, support, rounds=L, mode="bundle")
            assert np.array_equal(rerun, owner), (layer, "exchange owner")
        record = audit_and_count(routes, owner, support, home)
        record.update(archived_plan=provenance(path), fresh_sampled_support_exact=True,
            residency_diagnostics=diagnostics,
            replay_kind="fresh CPU full exchange replay equal archive" if optimize else "archived L=4 owner independently audited; native seed and sampled support freshly replayed",
            fresh_exchange_exact=True if optimize else None)
        if optimize is not None:
            record["exchange_details"] = exchange_details
        row["methods"]["sample256_g2_bundle"] = record
        sample = np.concatenate((routes.reshape(-1, K)[ids], fresh["native_rt0"].owner.reshape(-1, K)[ids]), axis=1).astype(np.int32)
        row["sample_routes_and_rt0_owners"] = dict(shape=list(sample.shape), dtype="int32", sha256=digest_array(sample),
            reconstruction="concatenate routes.reshape(N,K)[sample_ids] and rt0_owner.reshape(N,K)[sample_ids] along columns")
        for method, record in row["methods"].items():
            old = old_row["methods"][method]["metrics"]
            assert record["totals"]["remote_token_destinations"] == old["remote_token_destinations"]
            assert record["totals"]["active_extra_experts"] == sum(old["active_extra_experts"])
            assert [r["dispatch_tx_tokens"] for r in record["per_rank"]] == old["source_remote_payloads"]
            assert [r["dispatch_rx_tokens"] for r in record["per_rank"]] == old["destination_remote_payloads"]
        rows.append(row)
        print(json.dumps(dict(layer=layer, remote={m:r["totals"]["remote_token_destinations"] for m,r in row["methods"].items()},
            copies={m:r["totals"]["expert_copies_per_full_prefetch"] for m,r in row["methods"].items()})), flush=True)
    totals = {method: aggregate(rows, method) for method in METHODS}
    output = dict(schema_version=1, status="complete", scope="original eight logical ranks; CPU semantic counts only; no physical GPU timing or bandwidth estimate",
        protocol=dict(N=R*S, R=R, E=E, K=K, tokens_per_source=S, B=B, d=D, groups_per_class=2,
            sample_seed=SEED, L=L, layers=8, home_experts_per_rank=E//R, physical_gpu_mapping="none",
            native_rt0="popularity retention disabled; balancing-required remote copies retained",
            native_rt2="two source-popularity retention choices per rank before balancing; shared B=8 cap",
            optimized="rt0 seed, reciprocal sampled residency, four global bundle-exchange rounds"),
        environment=dict(python=platform.python_version(), numpy=np.__version__, numba_available=have_numba),
        sampling=dict(sample_ids=ids.tolist(), source_counts=np.bincount(ids//S, minlength=R).tolist(),
            original_evidence=provenance(evidence_path), exact_reproduction=True),
        copy_transfer_interpretation=dict(
            unit="one complete selected remote expert copied from its original home into one rank's extra slot",
            full_refresh="counts apply once per prefetch invocation per layer when every selected expert is refreshed",
            home_weights="44 original home experts per rank are parameters already resident; no extra copy charge",
            extra_slots="B additional prefetch slots per rank, backed by a process-global pool shared across layers, not persistent per-layer replicas",
            implementation="prefetch compacts every experts_to_copy>=0 entry and copies every tile; no ID/version cache test",
            optimizer_update="not necessarily one copy per optimizer update; microbatches, forward recomputation, prefetch invocation schedule and weight versions are external",
            reuse="zero-copy-cost reuse requires retained correct layer/expert/slot/version contents and explicitly skipping prefetch; not established by one router dump",
            shape="router dump contains no weights; counts deliberately do not assume H, intermediate size, dtype, gradients or bandwidth"),
        source_files=[provenance(ROOT / f) for f in ("experiments/eight_rank_estimate/traffic.py", "analysis/moonep_cpu.py",
            "analysis/vendor/residency.py", "analysis/vendor/bucket_optimizer.py", "inputs/MoonEP/moonep/api.py",
            "inputs/MoonEP/moonep/prefetch.py", "inputs/MoonEP/README.md")],
        original_results=provenance(archive_results_path), layers=rows, aggregate=totals,
        verified_totals=dict(remote=EXPECTED_REMOTE, expert_copies=EXPECTED_COPIES), failures=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), totals={m:r["totals"] for m,r in totals.items()})))


if __name__ == "__main__":
    main()
