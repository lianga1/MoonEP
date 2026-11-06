# 32-SM planner: frozen 03 results

All formal records, original GPU traces, output hashes, CPU-oracle hashes, and test exit records passed independent validation.

Hardware: one omni-dev host, 8 × NVIDIA H200 (132 SM/GPU). Workload: 134,144 tokens total, E=352, K=8; sample size 256, candidate cap 128, four exchange rounds.

Latency in microseconds. Each layer uses the maximum of eight rank medians (10 samples, 5 Graph replays/sample); the headline is the mean of eight layers.

| Variant | Mean GPU µs | Mean host µs | Change vs bracketing baseline | Max launch CTAs | Peak concurrent launched CTAs | Kernels/replay/rank |
|---|---:|---:|---:|---:|---:|---:|
| baseline_before_03 | 686.22 | 698.51 | +0.04% | 1584 | 1584 | 89 |
| bounded32_03 | 868.30 | 874.89 | +26.59% | 32 | 32 | 104 |
| fused32_03 | 895.52 | 902.09 | +30.56% | 32 | 32 | 64 |
| fused48_03 | 809.00 | 815.67 | +17.94% | 48 | 48 | 64 |
| fused64_03 | 777.98 | 784.63 | +13.42% | 64 | 64 | 64 |
| fused96_03 | 705.50 | 712.29 | +2.85% | 96 | 96 | 64 |
| fused128_03 | 699.04 | 705.72 | +1.91% | 128 | 128 | 64 |
| baseline_after_03 | 685.63 | 699.15 | -0.04% | — | — | — |

Bracketing baseline: 685.92 µs. Before→after drift: -0.086%. Positive change means slower. This arithmetic reference does not remove drift or provide a confidence interval.

| Layer | Before | Bounded 32 | Fused 32 | Fused 48 | Fused 64 | Fused 96 | Fused 128 | After |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 703.76 | 879.43 | 921.55 | 811.48 | 791.31 | 710.85 | 702.72 | 705.57 |
| 1 | 688.21 | 887.44 | 914.23 | 825.42 | 792.08 | 715.11 | 708.97 | 687.14 |
| 2 | 684.14 | 858.71 | 882.36 | 802.25 | 768.20 | 701.42 | 694.64 | 683.53 |
| 3 | 664.82 | 840.44 | 872.27 | 788.91 | 764.41 | 686.54 | 676.85 | 663.34 |
| 4 | 697.20 | 878.32 | 911.81 | 836.57 | 791.51 | 726.48 | 715.61 | 695.80 |
| 5 | 690.42 | 877.24 | 893.45 | 809.94 | 776.40 | 707.22 | 702.25 | 689.52 |
| 6 | 688.92 | 869.11 | 881.90 | 802.40 | 772.78 | 702.23 | 701.03 | 686.02 |
| 7 | 672.25 | 855.70 | 886.57 | 795.00 | 767.19 | 694.12 | 690.22 | 674.11 |

All variants reproduce the previous sampled planner exactly: all nine output arrays on every rank, expert support, selected candidate IDs, greedy history, owners, addresses, and source metadata. No formal run fell back.

Across the eight layers, remote token-destinations: 5,054,165 native MoonEP → 3,887,660 sampled planner, a 23.08007% reduction. Expert copies: 71 native → 434 sampled (+363); the SM/fusion changes add zero copies versus the previous sampled planner.

| Layer | Native remote destinations | Sampled remote destinations | Reduction | Native copies | Sampled copies | Greedy selections |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 621,511 | 533,736 | 14.12% | 11 | 61 | 30 |
| 1 | 643,011 | 485,678 | 24.47% | 7 | 55 | 29 |
| 2 | 635,585 | 465,156 | 26.81% | 7 | 52 | 29 |
| 3 | 623,501 | 476,656 | 23.55% | 10 | 50 | 26 |
| 4 | 634,668 | 483,420 | 23.83% | 8 | 56 | 32 |
| 5 | 625,260 | 487,250 | 22.07% | 9 | 50 | 32 |
| 6 | 635,436 | 479,503 | 24.54% | 9 | 57 | 31 |
| 7 | 635,193 | 476,261 | 25.02% | 10 | 53 | 28 |

Correctness suites:

- config: 24 passed per process; 1 process summaries; exit 0.
- components: 9 passed per process; 1 process summaries; exit 0.
- public: 15 passed per process; 8 process summaries; exit 0.
- compatibility: 5 passed per process; 8 process summaries; exit 0.

Evidence limits:

- Planner-only latency; excludes expert weights, token activations, gradients, GEMM, optimizer, histogram building, allocation, JIT and audits.
- Communication reduction is exact route-plan accounting. This cohort does not measure end-to-end training speed or weight/gradient payload costs.
- SM upper bound uses conservative concurrent launched CTA counts from raw GPU trace intervals. It does not pin SM IDs or measure achieved occupancy.
- Budget traces cover layer 0 on all 8 ranks and 3 Graph replays; full-output and CPU-oracle checks cover all 8 layers.
- Runs are sequential with bracketing baselines, not randomized repeated cohorts or confidence intervals.
- Exploratory probes are retained verbatim and excluded from every formal timing table.
- Private router NPZ files are omitted; their original SHA-256 and independent CPU array hashes are retained.
