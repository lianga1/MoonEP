# Shared-subset GPU planner prototypes

These files implement eight **virtual routing ranks on one GPU**. They do not invoke remote dispatch, copy expert weights, or generate the final MoonEP communication plan. On 2026-11-02, the exact sequential, simultaneous flat-list and simultaneous indexed paths compiled and passed all eight real-trace layers on the requested h100-omni single H100 (PyTorch 2.13.0+cu130, CUDA 13.0, sm90). All 24 outputs matched their independent CPU targets element by element, including CUDA Graph replay. These correctness runs intentionally recorded no performance numbers; use the common benchmark harness for latency comparisons.

## API

```python
runner = SubsetRunner(routes_i32_cuda, support_bool_cuda, h=2, g=8)
runner.prepare(seed_i32_cuda)       # GPU mining, shortlist, data layout conversion
owner = runner.run(seed, rounds=4) # exact sequential-pair CPU semantics
owner = runner.run_parallel(seed)  # new simultaneous token-first schedule, policy=1
owner = runner.full(seed)          # prepare + exact sequential exchange
owner = runner.full_parallel(seed) # prepare + token-first simultaneous exchange
runner.prepare_indexed(seed)       # prepare + GPU template lookup construction
owner = runner.run_indexed(seed)   # same policy1 result, direct lookup proposals
owner = runner.full_indexed(seed)  # includes mining and lookup construction
```

Fixed shape contract: source rank count 8, top-k 8 distinct experts/token, global experts 352, at most 52 resident experts per destination (44 home plus 8 extras). `h` is 1 or 2; `g` is 1..8. Tokens per source may vary provided every source has the same count. Owner/routing arrays must be contiguous CUDA int32; support is contiguous bool. All workspaces are allocated in the constructor; all timed methods only enqueue fixed GPU kernels and CUB operations, with no host scalar read or allocation.

## Implemented memory/computation changes

* One-time AoS-to-SoA transform: subsequent lanes access consecutive tokens at a fixed top-k position, giving coalesced loads. Output is converted back to original layout.
* Expert-subset statistics are indexed in **destination resident-slot coordinates**: 52 + 52² entries per class rather than 352 + 352². Dense counters shrink from 127,238,144 to 2,822,144 bytes. Selection converts IDs back before comparison, retaining original expert-ID tie order.
* Warp-combined count updates use `__match_any_sync` to merge simultaneous equal histogram keys.
* Mining and shortlist selection run on GPU. Each top-g selection thread reads its slice once and keeps g local candidates, then the CTA merges winners. No CPU candidate table is assumed in `full()`.
* Templates are staged in CTA shared memory; a token's eight experts and current owners are loaded once for the proposal pass, reused across candidate checks.
* The indexed simultaneous variant additionally enumerates only the token's own eight singletons and at most 28 same-owner expert pairs. It checks executable destination ranks and directly queries a 705,536-byte table, avoiding scans of unrelated templates. Every table entry stores the original template's order, so equal-score proposals retain the flat-list algorithm's tie behavior. The table is generated from GPU-mined templates; `prepare_indexed` includes its zeroing and construction, while the flat variant does not pay this cost.
* For mixed size-1/size-2 balancing, the CPU dynamic program is replaced by an exact constant-work quota formula for any target volume. The formula, including DP tie order, passed 5,376 small exhaustive cases. Parallel reductions choose the best balanced volume, minimizing volume on gain ties.
* CUB radix sorting provides stable acceptance order. Exact mode sorts `(category, earliest feasible template, token)` to match CPU touched-list semantics. Simultaneous mode sorts `(category, token)`.

## Semantics and scheduling

Exact mode retains the previous 28 sequential card-pair schedule and rotates its start by 7 pairs per round. There are 112 sort/match/commit phases over four rounds. It eliminates repeated posting-list scans but still reloads token state between pairs. It must match `outputs/pruned_subset_indexed/layerXX_h2_g8.npz` bitwise for h2/g8.

The new simultaneous mode considers all directions in one pass per round, emits at most one proposal/token, balances each card-pair independently, then commits every accepted token concurrently. Thus only four sort/match/commit phases remain. This changes the schedule: quality must be independently measured. Policy 1 uses rotating token/round-dependent size and direction tie preferences, analogous to the earlier bucket method; policy 0 is a weaker first-best/early-exit ablation.

For T tokens, K selected experts/token, R ranks and M shortlisted directed templates, the flat simultaneous proposal pass does O(T M K) worst-case checks per round. The indexed pass does O(T K² R) worst-case work, with same-owner and resident-destination pruning before lookup, and no M factor. Its lookup construction costs O(R² B² + M) including clearing, where B=52 is resident capacity. These counts exclude shared sorting and quota selection and do not by themselves imply a latency improvement.

Policy 1 CPU reference (`subset_reference.py`) on all eight trace layers gives 3,776,109 remote token-destination payloads, a 25.28718% reduction from native rt0. Every rank remains at 134,144 compute pairs, every execution is resident, and exact recomputed gains equal predicted gains. The flat and indexed GPU paths now both match that reference exactly. Exact sequential mode's CPU/GPU result is 26.81848%. These percentages describe replayed communication counts, not measured network transfer or end-to-end MoE latency.

## Validation

* `subset_formula_check.py`: 5,376 combinations of small bounded counts and target volumes, compared with CPU bounded DP including exact quota choices.
* `subset_reference.py`: complete CPU equivalent of policy 1; saves expected per-layer owners.
* `subset_smoke.py --correctness-only`: H100 compilation, element-by-element checks of all three variants against both CPU targets, and CUDA Graph replay consistency. All 24 layer/variant combinations passed. Evidence: `outputs/h100_subset_validation/subset_smoke_results_02.json` and `subset_smoke_02.log`. Omitting this flag additionally records full-component event timing; the accepted comparison should use the common interleaved benchmark harness.

CPU-generated routing inputs and support are supplied before timing. `prepare` does not choose new expert residency: it mines shared-subset movement templates under fixed resident experts. End-to-end timing must still add initial MoonEP allocation/residency generation and final physical address/dedup construction as appropriate. Never compare this single-device virtual-rank timing with eight-GPU native timing as if hardware and communication semantics matched.

## Direct lookup and parallel-prefix update

`prepare_indexed` constructs a 705,536-byte GPU table of selected directed expert subsets; `run_indexed` enumerates only a token's own singletons/same-owner pairs. Template order is retained for exact policy-1 ties. `full_indexed` includes mining and table construction. Both simultaneous paths now use a 256-thread, three-chunk CUB BlockScan for the 768 category offsets, replacing the single-thread loop. The independent layer-0 profiler saw this kernel fall from 29.23 to 2.44 us/call; profiler values are diagnostic, not the accepted benchmark.

Final eight-layer mean-of-medians Graph timing: original bundle 321.912 us; 8-lane bundle 266.604 us; exact subset prepare+exchange 6511.256 us; simultaneous flat 913.990 us; indexed 787.682 us. Fixed residency selection and real dispatch are excluded. The raw native MoonEP R1 control failed cuMulticastCreate initialization; the 228.356 us logical-R8 CUDA port must not be called native MoonEP latency.
