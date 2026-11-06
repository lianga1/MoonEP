"""Verify packaged integration evidence and recompute its performance summary."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def sha(data):
    return hashlib.sha256(data).hexdigest()


def measurement(value):
    raw = value["raw"]
    assert len(raw) == 10 and all(row["replays"] == 5 for row in raw)
    assert all([r["rank"] for r in row["per_rank"]] == list(range(8)) for row in raw)
    ms = max(statistics.median(row["per_rank"][rank]["cuda_ms"] for row in raw)
             for rank in range(8))
    assert abs(ms - value["max_rank_median_cuda_ms"]) < 1e-12
    return ms * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true",
                        help="also require current checkout source hashes to match the tested sources")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = Path(__file__).resolve().parents[3]
    manifest = json.loads((here / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha((here / name).read_bytes()) == expected, name
    for name, expected in manifest["uncompressed_sha256"].items():
        assert sha(gzip.decompress((here / name).read_bytes())) == expected, name
    stages = ("host_config", "selector", "cap128_public", "compatibility_public",
              "benchmark_cap0", "benchmark_cap128")
    assert all((here / f"{stage}_01.exit").read_text().strip() == "0" for stage in stages)
    assert (here / "complete_01").read_text().strip()
    logs = {stage: (here / f"{stage}_01.log").read_text() for stage in stages}
    assert logs["host_config"].count("18 passed") == 1
    assert logs["selector"].count("3 passed") == 1
    assert logs["cap128_public"].count("5 passed") == 8
    assert logs["compatibility_public"].count("5 passed") == 8
    result = {"scope": "fresh integrated public API; eight physical and logical ranks; four exchange rounds",
              "timing": "max of eight rank medians of ten samples, five graph replays each; mean of the same eight layers",
              "test_cases": {"host_configuration": 18, "single_gpu_selector": 3,
                  "cap128_public_cases": 5, "existing_compatibility_cases": 5,
                  "distributed_ranks_per_case": 8,
                  "note": "five distributed cases executed on eight ranks are five cases, not forty independent cases"},
              "all_six_exit_codes_zero": True, "variants": {},
              "current_checkout_sources_checked": args.check_sources}
    records = {}
    for cap in (0, 128):
        data = json.loads(gzip.decompress((here / f"benchmark_cap{cap}_01.json.gz").read_bytes()))
        records[cap] = data
        assert data["status"] == "complete" and not data["failures"]
        assert [x["layer"] for x in data["layers"]] == list(range(8))
        meta = data["metadata"]
        assert (meta["R"], meta["N"], meta["E"], meta["K"], meta["B"]) == (8, 134144, 352, 8, 8)
        assert meta["planner_config"] == dict(sample_size=256, groups_per_class=2, rounds=4,
                                               sample_seed=20261026, candidate_limit=cap)
        assert len({x["uuid"] for x in meta["identities"]}) == 8
        assert meta["oracle_sha256"] == sha((here.parent / "cpu_array_hashes.json").read_bytes())
        if args.check_sources:
            for name, expected in meta["source_files"].items():
                assert sha((repo / name).read_bytes()) == expected, name
        native_us, sampled_us, native_remote, remote, copies = [], [], 0, 0, 0
        for row in data["layers"]:
            assert row["graph_outputs_exact"] and row["all_rank_history_exact"] and row["cpu_oracle_checked"]
            assert row["native_audit"]["passed"] and row["audit"]["passed"]
            assert not row["audit"]["fell_back_to_native"]
            assert row["audit"]["status_by_rank"] == [0] * 8
            native_us.append(measurement(row["variants"]["native"]))
            sampled_us.append(measurement(row["variants"]["sampled"]))
            native_remote += row["native_audit"]["remote_token_destinations"]
            remote += row["audit"]["remote_token_destinations"]
            copies += row["audit"]["expert_copies"]
        item = dict(native_mean_us=statistics.mean(native_us), sampled_mean_us=statistics.mean(sampled_us),
                    native_us=native_us, sampled_us=sampled_us, native_remote=native_remote,
                    remote=remote, copies=copies, reduction_percent=100 * (1 - remote / native_remote))
        supplied = json.loads((here / f"provided_summary_cap{cap}_01.json").read_text())
        for name in ("native_mean_us", "sampled_mean_us", "native_remote", "remote", "copies", "reduction_percent"):
            assert abs(item[name] - supplied[name]) < 1e-9, name
        result["variants"][str(cap)] = item
    assert records[0]["metadata"]["source_files"] == records[128]["metadata"]["source_files"]
    assert records[0]["metadata"]["identities"] == records[128]["metadata"]["identities"]
    historical = json.loads(gzip.decompress((here.parent / "full_active_rounds4_01.json.gz").read_bytes()))
    for cap in (0, 128):
        for fresh, prior in zip(records[cap]["layers"], historical["layers"]):
            assert fresh["fixture_sha256"] == prior["fixture_sha256"]
            old_audit = prior["variants"][str(cap)]["audit"]
            for key in ("remote_token_destinations", "expert_copies", "owner_sha256"):
                assert fresh["audit"][key] == old_audit[key], (cap, fresh["layer"], key)
    result["same_source_manifest_and_devices"] = True
    result["historical_fixture_traffic_copies_and_owner_hashes_match_each_layer"] = True
    baseline, limited = result["variants"]["0"], result["variants"]["128"]
    result["planner_time_reduction_percent"] = 100 * (1 - limited["sampled_mean_us"] / baseline["sampled_mean_us"])
    result["retained_communication_savings_percent"] = 100 * (limited["native_remote"] - limited["remote"]) / (baseline["native_remote"] - baseline["remote"])
    result["limitations"] = ["one same-environment comparison batch, not an order-randomized repeated study",
        "no expert/token/gradient payload transport, GEMM, optimizer or full training-step latency"]
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
