"""Plot controlled rail imbalance; no measured layer-latency claims."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "epmoe_matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("balanced", "fanout_skew_1p5", "fanout_skew_2p0")
COLORS = ("#3A7F94", "#D99A32", "#C95250")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "outputs/multinode_rail/scenarios/manifest.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/multinode_rail/rail_comparison")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    lookup = {r["scenario"]: r for r in manifest["scenarios"]}
    rows = [lookup[n] for n in SCENARIOS]
    stats = [r["statistics"] for r in rows]
    for key in ("rdma_token_copies", "nvlink_token_copies", "remote_pairs",
                "expert_pair_counts_sha256"):
        assert len({s[key] for s in stats}) == 1, f"controlled comparison changed: {key}"
    assert all(s["pair_loads_per_rank"] == [32768] * 32 for s in stats)
    copies = np.asarray([s["rail_rdma_token_copies"] for s in stats])
    ratios = [s["rail_statistics"]["max_over_mean"] for s in stats]
    # One payload copy is one hidden vector. Full-duplex TX and RX service
    # bounds are separate; their maximum is the port bottleneck bound.
    payload_bytes = 7168 * 2
    assumed_bandwidth_bytes_s = 40e9
    busiest_nic = np.asarray([
        max(np.max(s["nic_tx_token_copies"]), np.max(s["nic_rx_token_copies"]))
        for s in stats
    ])
    lower_bounds_ms = busiest_nic * payload_bytes / assumed_bandwidth_bytes_s * 1e3

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#374151", "text.color": "#17212F",
        "xtick.color": "#4B5563", "ytick.color": "#4B5563",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 7.0), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=0.07, right=0.975, top=0.76, bottom=0.27, wspace=0.28)
    fig.suptitle("Rail imbalance at equal total communication volume", fontsize=19, y=0.95, fontweight="bold")
    fig.text(0.5, 0.887, "Constructed 4-node / 32-GPU workloads  |  131,072 tokens  |  top-8 routing", ha="center", fontsize=12, color="#4B5563")

    ax = axes[0]
    x = np.arange(copies.shape[1])
    width = 0.235
    for i, color in enumerate(COLORS):
        ax.bar(x + (i - 1) * width, copies[i], width=width, color=color,
               label=f"Rail max/mean = {ratios[i]:.1f}", zorder=3)
    ax.set_title("RDMA token copies per rail, summed across 4 nodes", fontsize=12.3, pad=17, loc="left")
    ax.set_ylabel("Token copies (absolute count)")
    ax.set_xlabel("Rail / source local-rank index", labelpad=12)
    ax.set_xticks(x)
    ax.set_ylim(0, 55000)
    ax.set_yticks(np.arange(0, 50001, 10000))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.25), ncol=3, frameon=False, fontsize=10.0,
              handlelength=1.1, handletextpad=0.5, columnspacing=1.1)

    ax = axes[1]
    bars = ax.bar(np.arange(3), lower_bounds_ms, width=0.58, color=COLORS, zorder=3)
    for bar, ms in zip(bars, lower_bounds_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, ms + 0.12, f"{ms:.3f} ms",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Busiest NIC: RDMA payload service bound", fontsize=12.3, pad=17, loc="left")
    ax.set_ylabel("Lower bound (ms)")
    ax.set_xticks(np.arange(3), [f"{r:.1f}" for r in ratios])
    ax.set_xlabel("Rail max/mean", labelpad=12)
    ax.set_ylim(0, 5.2)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.text(0.5, -0.29, "7,168 hidden elements · BF16 · assumed 40 GB/s per NIC",
            ha="center", va="center", transform=ax.transAxes, fontsize=9.7, color="#4B5563")

    rdma = stats[0]["rdma_token_copies"]
    nvl = stats[0]["nvlink_token_copies"]
    fig.text(0.07, 0.092,
             f"All three cases: {rdma:,} total RDMA copies · {nvl:,} total NVLink copies · 32,768 execution pairs per GPU",
             fontsize=11.2, fontweight="bold")
    fig.text(0.07, 0.048,
             "Right panel: illustrative payload service lower bound, not measured layer latency. Bound = max(NIC TX bytes, NIC RX bytes) / bandwidth.",
             fontsize=10, color="#596273")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(args.out.with_suffix(suffix), dpi=180)
    plt.close(fig)
    print(json.dumps({"png": str(args.out.with_suffix(".png")),
                      "svg": str(args.out.with_suffix(".svg")),
                      "busiest_nic_payload_service_lower_bound_ms": lower_bounds_ms.tolist(),
                      "status": "illustrative payload lower bounds, not measured layer latency"}, indent=2))


if __name__ == "__main__":
    main()
