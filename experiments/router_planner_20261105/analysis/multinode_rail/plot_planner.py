"""Plot exact CPU-replay communication metrics, not hardware speedups."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/multinode_rail/planner_sparse")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/multinode_rail/planner_comparison")
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text())
    rows = {(r["scenario"], r["method"]): r for r in summary["rows"]}
    names = ["fanout_skew_1p5", "fanout_skew_2p0", "trace_lift_layer00", "trace_lift_layer07"]
    for name in names:
        for method in ("seed", "rail_b8_r1"):
            detail = json.loads((args.input / f"{name}_{method}.json").read_text())
            assert detail["traffic"]["pair_loads_per_rank"] == [32768] * 32
            assert max(detail["reserved_extra_per_rank"]) <= 8
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#17212F",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    seed_color, planned_color, nvl_color = "#AAB6C2", "#287F8F", "#D79A36"
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 6.8),
                             gridspec_kw={"width_ratios": [1.05, 1.05, 1.25]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.73, bottom=0.27, wspace=0.31)
    fig.suptitle("CPU replay: one-round planner communication metrics", fontsize=20,
                 y=0.955, fontweight="bold")
    fig.text(0.5, 0.89, "4 nodes × 8 GPUs  |  131,072 tokens  |  top-8  |  32,768 execution pairs per GPU preserved",
             ha="center", fontsize=12, color="#475569")
    for ax, name in zip(axes[:2], names[:2]):
        details = [json.loads((args.input / f"{name}_{method}.json").read_text())
                   for method in ("seed", "rail_b8_r1")]
        values = [d["traffic"]["rail_rdma_token_copies"] for d in details]
        before, after = [rows[(name, m)]["rail_max_over_mean"] for m in ("seed", "rail_b8_r1")]
        x, width = np.arange(8), 0.36
        ax.bar(x - width / 2, values[0], width=width, color=seed_color, label="Seed", zorder=3)
        ax.bar(x + width / 2, values[1], width=width, color=planned_color, label="B8 · 1 round", zorder=3)
        ax.set_title(f"Rail max/mean: {before:.1f} → {after:.3f}", fontsize=13, pad=19, loc="left")
        ax.set_xticks(x)
        ax.set_xlabel("Rail / source local rank", labelpad=11)
        ax.set_ylim(0, 55000)
        ax.set_yticks(np.arange(0, 50001, 10000))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.025, -0.23), ncol=2,
                  frameon=False, fontsize=10, handlelength=1.5)
    axes[0].set_ylabel("RDMA token copies per rail\n(summed across 4 nodes)", labelpad=10)

    ax = axes[2]
    labels = ["Fanout 1.5", "Fanout 2.0", "Trace lift L00", "Trace lift L07"]
    nic = np.asarray([rows[(n, "rail_b8_r1")]["max_nic_copies_reduction_pct"] for n in names])
    nvl = np.asarray([rows[(n, "rail_b8_r1")]["nvlink_copies_reduction_pct"] for n in names])
    y, height = np.arange(len(names)), 0.33
    ax.barh(y - height / 2, nic, height=height, color=planned_color, label="Busiest NIC", zorder=3)
    ax.barh(y + height / 2, nvl, height=height, color=nvl_color, label="Total NVLink", zorder=3)
    for at, ns, vs in zip(y, nic, nvl):
        ax.text(ns + 0.3, at - height / 2, f"{ns:.2f}%", va="center", fontsize=10, fontweight="bold")
        ax.text(vs + 0.3, at + height / 2, f"{vs:.2f}%", va="center", fontsize=10)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 18.6)
    ax.set_xticks([0, 5, 10, 15])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}%"))
    ax.set_title("Payload reduction vs seed", fontsize=13, pad=19, loc="left")
    ax.set_xlabel("Reduction in token-copy count", labelpad=11)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.23), ncol=2,
              frameon=False, fontsize=10, handlelength=1.5)

    fig.text(0.06, 0.105,
             "Extra experts per GPU: seed 0–2; planned ≤8 (B8). The comparison uses different actual replica counts.",
             fontsize=11, fontweight="bold")
    fig.text(0.06, 0.055,
             "Token payload only; weight/gradient transfers excluded. Pair counts are fixed, but padding may change. GPU planner latency is unmeasured.",
             fontsize=10.3, color="#596273")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    plt.close(fig)
    print(json.dumps({"png": str(args.output.with_suffix(".png")),
                      "svg": str(args.output.with_suffix(".svg")),
                      "scenarios": names,
                      "busiest_nic_reduction_pct": nic.tolist(),
                      "nvlink_total_reduction_pct": nvl.tolist()}, indent=2))


if __name__ == "__main__":
    main()
