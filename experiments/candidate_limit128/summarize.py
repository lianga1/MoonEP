"""Verify preserved evidence and recompute cap-128 vs unlimited statistics."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest((root / name).read_bytes()) == expected, name
    raw = gzip.decompress((root / "full_active_rounds4_01.json.gz").read_bytes())
    assert digest(raw) == manifest["raw_uncompressed_sha256"]
    data = json.loads(raw)
    assert data["status"] == "complete" and not data["failures"]
    assert [row["layer"] for row in data["layers"]] == list(range(8))
    metadata = data["metadata"]
    assert metadata["rounds"] == 4 and metadata["R"] == 8
    assert len({item["uuid"] for item in metadata["identities"]}) == 8
    assert (root / "full_active_rounds4_01.exit").read_text().strip() == "0"
    cpu = json.loads((root / "cpu_summary_02.json").read_text())
    result = {"scope": "historical pre-integration wrapper, eight physical GPUs",
              "source_sha256": digest(raw), "variants": {}}
    for cap in (0, 128):
        layer_times = []
        remote = copies = 0
        for layer in data["layers"]:
            item = layer["variants"][str(cap)]
            assert item["graph_outputs_exact"] and item["cpu_oracle_checked"]
            assert item["audit"]["passed"] and not item["audit"]["fell_back_to_native"]
            samples = item["timing"]["raw"]
            assert len(samples) == 10 and all(row["replays"] == 5 for row in samples)
            rank_times = [statistics.median(row["per_rank"][rank]["cuda_ms"]
                                            for row in samples) for rank in range(8)]
            measured = max(rank_times)
            assert abs(measured - item["timing"]["max_rank_median_cuda_ms"]) < 1e-12
            layer_times.append(measured * 1000)
            remote += item["audit"]["remote_token_destinations"]
            copies += item["audit"]["expert_copies"]
        result["variants"][str(cap)] = dict(full_planner_us=statistics.mean(layer_times),
            per_layer_full_planner_us=layer_times, remote_token_destinations=remote,
            expert_copies=copies, reduction_percent=100 * (1 - remote / cpu["native_remote"]))
    baseline, limited = result["variants"]["0"], result["variants"]["128"]
    result["planner_time_reduction_percent"] = 100 * (1 - limited["full_planner_us"] / baseline["full_planner_us"])
    result["retained_communication_savings_percent"] = 100 * (
        cpu["native_remote"] - limited["remote_token_destinations"]) / (
        cpu["native_remote"] - baseline["remote_token_destinations"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
