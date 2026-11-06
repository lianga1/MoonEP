# Fresh integrated cap128 verification

This directory records the production `Buffer`/`SampledPlannerConfig` path with
`candidate_limit=128`, `rounds=4`, `B=8`, and eight physical/logical ranks. The
unlimited control uses `candidate_limit=0` in the same code and environment.
It is separate from the historical wrapper measurements in the parent folder.

The eight-H200 comparison uses the same eight trace layers, N=134,144,
E=352/K=8, 256 global samples, two groups per class and seed 20261026. Ten timing
samples each contain five warmed CUDA Graph replays; the reported layer time
is the maximum of the eight rank medians. The final value averages all eight
layers.

| Fresh integrated comparison | Unlimited | Limit 128 |
|---|---:|---:|
| Complete planner mean | 815.855619 µs | 684.846398 µs |
| Native MoonEP mean in the same run | 68.581199 µs | 68.428000 µs |
| Remote token destinations, eight layers | 3,791,665 | 3,887,660 |
| Reduction relative to native MoonEP | 24.9794% | 23.0801% |
| Extra expert copies, eight layers | 507 | 434 |

The measured planner-time reduction is 16.0579%; cap128 retains 92.3964% of
the unlimited configuration's communication savings. Native remote traffic
is 5,054,165 visits. Each layer's fixture hash, traffic, copies and final owner
hash matches the corresponding historical result. All post-Graph, all-rank
history, CPU-oracle and physical-layout checks passed without normal fallback.

Validation includes 18 host-configuration cases, three single-GPU selector
cases, five cap128 distributed public-interface cases and five existing
compatibility cases. Each distributed case runs jointly on all eight ranks:
eight `5 passed` log lines represent **five distributed cases**, not forty
independent cases. The selector tests cover the delivery shared-cache kernel,
small-model candidate-cap clipping and the generic shared-memory fallback.
Public tests include changed Graph inputs, configuration agreement, forced
fallback/recovery, saved plans and weighted forward/backward behavior.

All six stage exit codes are zero. The benchmark source manifest covers the
exact runtime and measurement code; GPU identities and software versions are
retained in the raw JSON metadata.

## Preserved files

- `benchmark_cap0_01.json.gz`, `benchmark_cap128_01.json.gz`: unchanged original
  benchmark JSON bytes, compressed with deterministic gzip.
- `*.log`, `*.exit`, `complete_01`: unchanged original logs and completion
  evidence, including the launcher log.
- `original_validation.sh.gz`: unchanged original validation script. Its
  deployment-specific working directory is historical provenance, not a
  required path for another installation.
- `run_validation.sh`: portable equivalent with explicit fixture and output
  directories. It refuses to overwrite an existing output directory.
- `provided_summary_cap0_01.json`, `provided_summary_cap128_01.json`: the
  separately generated integration summaries, preserved unchanged.
- `manifest.json`: SHA256 of packaged evidence and scripts, plus the original
  decompressed JSON/script hashes. The historical parent manifest is unchanged.
- `summarize.py`, `summary.json`: independently recomputed aggregate results
  from the raw per-rank samples, with completion, shape and comparison checks.

Verify offline without CUDA or third-party Python packages:

```bash
python experiments/candidate_limit128/integration/summarize.py
# Also verify that the current checkout has the exact tested runtime sources:
python experiments/candidate_limit128/integration/summarize.py --check-sources
```

Run fresh tests on a suitable, available eight-GPU host, from the repository:

```bash
bash experiments/candidate_limit128/integration/run_validation.sh \
  /path/to/router-fixtures /path/to/new-results
```

The caller must provide the authorized original fixtures; no private router
arrays are packaged. `--check-sources` is intended for the tested commit and
may fail after legitimate later code changes; the raw evidence can still be
verified without that option.

This is one same-environment comparison batch, not an order-randomized
multi-run study. Planner timing includes native seed planning, live sampling,
GPU candidate pruning, expert selection, four exchange rounds, layout,
validation and metadata publication. It excludes input histogram construction,
JIT, allocation, audits, expert/token/gradient payload transport, GEMM and the
optimizer. Numerical forward/backward tests do not establish expert-gradient
aggregation performance or a full training-step speedup.
