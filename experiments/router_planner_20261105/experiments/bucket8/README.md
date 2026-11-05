# Top-8 GPU bucket exchange and eight-GPU count coordination

This experiment is the fixed-residency exchange component. Inputs already contain the native MoonEP seed and a chosen support matrix. It does not include seed/support preparation, route distribution, dispatch, combine, expert weight copies or gradient traffic.

## Measured status, 2026-10-26

All three modes have now run on omni-dev H200 hardware with CUDA Graph: one physical GPU for `single`, eight physical GPUs for `local` and `coordinated`. Across 8 real layers, 3 support methods and one tail per mode, all 75 cases matched their CPU owners exactly and passed residency, load and slot checks. GPU work has finished and resources were released.

With reciprocal support, the mean of eight per-layer median component times is **326.884 us single → 254.456 us coordinated**, a **22.16% latency reduction / 1.285x speedup**, preserving **24.6166%** remote-token-destination reduction versus the native seed. The latter uses summed counts over eight layers, not the arithmetic mean of layer percentages. Independent local matching takes **193.856 us**, but its reduction is **15.5456%**. These are fixed-support exchange measurements, not complete planner or training timings.

See [the measured report](../../outputs/gpu/BUCKET8_REPORT_ZH.md), [machine-readable aggregation](../../outputs/gpu/bucket8_summary.json), and raw files in `results/{single,local,coordinated}_graph.json`. `results/single.json` retains the eager tail smoke separately.

The CUDA proposal kernel retains **16 lanes per token**, with only lanes 0–7 loading actual top-k entries. All 16 lanes still cooperate on 64 directed rank pairs. This avoids changing the established half-warp ballot/shuffle contract when moving from top-16 to top-8.

Each round performs local proposals, a stable CUB radix sort and bucket counts. There are `8*8*2*(8+1)=1152` int32 counts: 4608 input bytes per source GPU. The coordinated version uses one `all_gather_into_tensor` per round, then every GPU derives global acceptance quotas and its source-prefix count. A local proposal's global ordinal is `prefix[source,bucket]+local_ordinal`. With source-ordered contiguous token shards this is exactly the global stable sort order. Four rounds require four metadata collectives; full owner maps are collected only for correctness, outside timing.

Modes:

- `single`: one physical GPU holds all eight logical source ranks and matches global buckets.
- `local`: eight GPUs each match only their own source shard. This can lose communication savings but preserves loads.
- `coordinated`: eight GPUs process source shards and match global buckets using the gathered counts. Output must match the single/global CPU reference pair by pair.

Prepare the 24 real input cases and tail fixture from the frozen CPU replays:

```bash
python experiments/bucket8/prepare_inputs.py
python experiments/bucket8/audit_protocol.py
```

The independent CPU protocol audit checks 40 rounds across real and random cases, including 1/2/19/23 tokens per source. It verifies the CUDA proposal scoring formula and count-prefix protocol against the existing CPU bucket implementation, plus residency and load preservation. It does **not** validate CUDA execution or performance.

On a confirmed available machine, first validate tails:

```bash
bash experiments/bucket8/run.sh single --tail-only --warmup 2 --reps 5
bash experiments/bucket8/run.sh local --tail-only --warmup 2 --reps 5
bash experiments/bucket8/run.sh coordinated --tail-only --warmup 2 --reps 5
```

Then omit `--tail-only` for all eight layers and `seed/visited/reciprocal` B8 supports. `--layers 0 2 5 7 --methods reciprocal` selects a subset. CUDA visibility is inherited, so single GPU tests can use an explicitly selected idle GPU. Distributed modes require eight visible GPUs; they do not allocate or reserve resources automatically.

Timing includes seed-to-owner copy and four exchange rounds. Both CUDA-event time and synchronized host wall time are reported, with the maximum over physical ranks for each distributed repetition. Compilation, input upload, support packing, full owner collection and correctness checks are outside timing. Default execution is eager: CUDA-event elapsed time can include host launch supply gaps and is not a pure kernel-duration sum. Each run preserves raw per-rank/per-repetition timings and source/input hashes.

Add `--graph` to capture the full run, including seed copy, four rounds and all four count AllGathers in coordinated mode. All ranks warm the same path before capture. Graph results are written separately as `single_graph.json`, `local_graph.json`, and `coordinated_graph.json`; eager results remain. Capture failures are retained as per-rank JSON and terminate the run instead of silently falling back. Compare graph with graph across modes, or eager with eager. All three Graph modes succeeded in the measured run above.

Original algorithm provenance: `experiments/moe_placement/hotcore128k/gpu/bucket_cuda.{cu,py}` in the earlier EP project. Adaptation and coordination are isolated in this directory; the original files are unchanged.
