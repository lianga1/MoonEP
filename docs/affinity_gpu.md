# GPU affinity planner

This fork adds `planning_backend="affinity_gpu"` to `moonep.Buffer`. The default
`"moon"` backend remains the upstream planner. No CPU solver, LP, or per-token
host decisions are used. The optimizer runs on EP rank 0; the native adapter
collects routing metadata and publishes one consistent plan to every EP rank.

## Installation and use

Install MoonEP's normal CUDA/PyTorch dependencies, then:

```bash
pip install -e '.[affinity]'
```

The optional `affinity` dependency installs Ninja. The CUDA source is included in
both source distributions and wheels. The first affinity `Buffer` construction
JIT-compiles `moonep/csrc/affinity_planner.cu` using the installed CUDA toolkit.
Compilation is cached by PyTorch and must happen before graph capture.

```python
import torch.distributed as dist
from moonep import Buffer

# Initialize the NCCL process group and select the local CUDA device first.
buffer = Buffer(
    S=12500, H=4096, K=8, E=256, num_ep_ranks=8,
    B=32, group=dist.group.WORLD,
    planning_backend="affinity_gpu", planning_sweeps=2,
)
hidden_nvs, weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
)
# Existing prefetch_weight, expert computation, combine and reduce_grad APIs
# consume the native plan. Backward can reuse this plan via dispatch(plan=plan).
buffer.destroy()  # before destroying the process group
```

All ranks in the EP process group must select the same backend, sweep count and
shape. `topk_experts_sk` is contiguous CUDA int32 `[S,K]`, with distinct IDs in
`[0,E)` for each token. `tokens_per_expert` is the corresponding local histogram.
Each process has the same `S`. The supported optimizer range is `2 <= R <= 16`,
`K <= 32`, `S*K <= 2097151`, `E <= 1024`, `E % R == 0`, and `0 < B <= E-E/R`. A CUDA device supporting
cooperative launch is required, alongside MoonEP's existing NVLink requirements.

Warm up fresh dispatch before capturing CUDA Graphs so that CuTe kernels and
NCCL collectives are initialized. Synchronous dispatch, `async_finish=True`,
plan reuse and EP subgroups use the existing Buffer interfaces. Concurrent calls
on the same Buffer are unsupported because its scratch buffers are reused.

## Algorithm

1. Run the original MoonEP planner to obtain an explicit `(token, expert)`
   assignment. All-gather source routing IDs and encoded destination rows.
2. On EP rank 0, decode the seed. For each rank, retain home experts and every
   remote expert required by the seed. Fill remaining residency slots with
   experts most frequently requested by tokens that already visit that rank.
   Residency does **not** imply that all possible pairs execute there.
3. Visit unordered rank pairs in diagonal groups (`a+b` constant). Pairs in
   one group are disjoint and can run concurrently. For each token, scan its
   `K` explicit assignments, consider moves whose expert resides on the other
   rank, and propose at most one candidate for that rank pair.
4. A move's exact reduction in remote token destinations is
   `gain = (old != source and old_count == 1)
         - (new != source and new_count == 0)`.
   Keep gain-1 and gain-0 candidates in each direction. Match equal counts in
   opposite directions, and require positive total gain for each accepted
   exchange. A gain-0 candidate only participates when paired with gain-1.
   Equal-count exchanges preserve the seed's exact per-rank pair load.
5. Repeat the fixed number of sweeps (default two). Pack explicit assignments
   into MoonEP's native execution rows, expert-copy slots, padded offsets,
   negative duplicate markers and source metadata. Broadcast the result and
   publish `src_info` through MoonEP's NVLink buffers before dispatch.

The persistent cooperative CUDA kernel uses shared resident masks, warp ballot
compaction, CTA queue prefix sums, and disjoint rank-pair teams. Candidate queues
are populated atomically; the result is not bitwise deterministic and need not
match the CPU greedy algorithm. It is a bounded greedy exchange search, not DP
or an optimal solver. Each accepted exchange reduces actual remote token
transfers, subject to fixed residency and exact load conservation.

If the original seed is not exactly balanced, its required replicas exceed the
slot budget, or packing exceeds row capacity, a device-side status selects the
original MoonEP plan instead. This fallback remains graph-compatible and is
reported by the benchmark. A fallback preserves the original planner's load;
it does not promise exact balance for a seed that was already imbalanced.
The caller remains responsible for valid routing IDs and histograms.

The native plan contains only **active** remote experts. Unused proposed
residency slots need not trigger weight copies. The benchmark reports active
replicas separately from the configured slot budget.

## Complexity and communication

Let `T=R*S` be global tokens, `K` top-k, `E` experts, `B` extra slots per rank,
and `I` exchange sweeps. The scan does `O(I*R^2*T*K)` work. Diagonal teams reduce
the serialized rank-pair stages to `I*(2R-3)` but do not remove that total work.
Resident selection uses `O(R*T*K + R*B*E)` work. Packing is `O(T*K + R*E)` with
the shared backend. Scratch storage includes `O(R*T + T*K + R*E)` plus standalone
seed histogram storage. These bounds describe the custom optimizer; native
end-to-end planning also includes the original planner and NCCL collectives.

Each process contributes `S*K` routing IDs and `S*K` encoded seed rows to two
all-gathers. The adapter broadcasts `T*K` final rows and smaller layout/status
buffers. Timing must include these operations. Reduced hidden-token traffic
can trade off against added planner latency and expert-weight traffic; this
benchmark does not claim an end-to-end training speedup.

`remote_token_destinations` counts distinct remote GPUs visited by each token,
summed over tokens. At fixed hidden width/dtype this is proportional to deduped
hidden dispatch payload; metadata and expert weights are excluded.
`N = T*K / all_token_destinations` includes local GPU destinations. Load spread
is `(max_load-min_load)/mean_load`, using explicit pair counts, not padding.

## Reproduce

From the repository root on eight NVLink-connected GPUs:

```bash
torchrun --standalone --nproc_per_node=8 -m benchmarks.bench_affinity_planner
```

This generates 100K tokens, 256 experts, top-8 and 32 extra slots, using a
Gaussian popularity curve whose endpoint ratio is approximately 7:1. It checks
native plan invariants, actual public dispatch/combine, saved-plan reuse,
replica capacity and load, then times both planners with the unchanged upstream
`benchmarks.bench_comm.time_gpu_op` CUDA Graph timer. Compilation and audits are
outside the measured region. Output is `affinity_results.json`.

For real selection data, supply a local NPZ with integer `routes` of shape
`[R,S,K]` (the first dimension must equal physical EP size):

```bash
torchrun --standalone --nproc_per_node=8 -m benchmarks.bench_affinity_planner \
  --replay /path/to/routes.npz --output replay_results.json
```

Routing traces and cluster access configuration are not required to install the
fork. The benchmark never silently repartitions an eight-rank trace as a
two-rank workload. Use `--slots 16`, `--sweeps 1`, or `--sweeps 0` for ablations;
inspect `fallback_to_moon` before interpreting the result.

Run integration and upstream regression tests:

```bash
torchrun --standalone --nproc_per_node=8 -m pytest -q tests/test_affinity_gpu.py
torchrun --standalone --nproc_per_node=8 -m pytest -q tests/test_planning.py
```

Integration tests cover padding, zero/two sweeps, more slots than home experts,
async dispatch/combine, plan reuse, CUDA Graph replay, a nonzero global subgroup
root and forced device-side fallback. Multi-GPU *sharding of the optimization*
is a separate experimental prototype and is not enabled by this backend.

## H200 validation (2026-10-10)

Tested on eight physical H200 GPUs with PyTorch
`2.7.0a0+ecf3bae40a.nv25.02`, CUDA 12.8 and CUTLASS DSL 4.4.2, based on upstream
MoonEP `2bd860b4dd083df62b79d5e916fca71ec5742228`.
Both runs used 100K global tokens, E=256, K=8, B=32 and two sweeps.
The figures below are medians of three batches of ten graph-captured iterations,
using the upstream timer's cross-rank mean. Initialization/JIT and audit time are
excluded; native seed planning, all-gathers, broadcasts and publication are included.

| Input | MoonEP planner | Affinity planner | Remote token destinations | Reduction | N (MoonEP → affinity) |
|---|---:|---:|---:|---:|---:|
| Real selection replay | 75.344 µs | 705.154 µs | 467,450 → 321,330 | 31.26% | 1.494 → 2.027 |
| Gaussian 7:1 sampling | 82.164 µs | 707.546 µs | 463,605 → 377,625 | 18.55% | 1.508 → 1.778 |

Every rank executed exactly 100,000 pairs in both runs: load spread was 0%, and
neither run fell back to MoonEP. Active remote replicas totaled 255 for real
replay (31–32 per rank), compared with 10 in MoonEP; synthetic data used 212,
compared with 38. Proposed residency is capped at 32 extra experts per rank;
only experts with assigned pairs appear in `experts_to_copy`.

The real-replay integrated planner is **9.36× slower** than the original planner
while reducing deduplicated remote hidden-token payload by 31.26%. Its 0.705 ms
runtime is below 1 ms on this workload; this is not a guarantee for other shapes,
and it is not a measured training speedup. Expert weight/gradient traffic and
FFN execution are excluded. Atomic candidate ordering can slightly change
communication totals between runs.

Validation performed on each of the eight processes:

- Four integration cases passed (public dispatch/combine, expert-row mapping,
  route weights, plan reuse, async execution, graph replay, subgroup and fallback).
- Upstream `tests/test_planning.py`: 18 passed, one skipped because the case
  requires at most two ranks.
- Source distribution and wheel built successfully; both contain the adapter
  and the CUDA source required for JIT compilation.

Machine-readable measurements: [real replay](benchmarks/h200_replay_100k.json)
and [synthetic](benchmarks/h200_synthetic_100k.json). The replay file is identified
by SHA-256 in the JSON; raw selection data is not redistributed in this fork.
