# Configurable sampled_group validation

Base commit: `45c2432a8da6f465028e833464f0ec25379cce0e`. Actual host: CI-情境智能 / omni-dev, eight distinct H200 GPUs. See [configuration and measured results](../../docs/sampled_configurable_validation.md).

- `evidence/`: raw logs, exit codes, full original-trace replay and eight-GPU planner timing, original/final runtime source manifests.
- `matrix_01_test_source.py`: original nine-case matrix before adding the E8/K8 boundary case. Seven runtime configurations and four initialization checks passed before the E4096/B512 native compile was deliberately stopped. The nonzero exit and reason remain intact.
- `matrix_02`: final ten runtime configurations and four initialization checks; 14 tests passed on every rank after bounding the large-B slot loop. Production code hashes are in `runtime_manifest_final.json`; the initial manifest retains the previous unrolled-loop version.
- `residency_cpu_checks.json`: CPU-only structural checks; kept distinct from GPU tests.
- `independent_readback_01.json`: independently recomputed test counts, source hashes, per-rank timing medians, original-trace ownership equality with the previous commit, and traffic/copy totals.
- Shell/Python runners retain the actual shared paths for provenance. Adapt those paths for another environment. The native `_C` binary was reused from the previous build because bindings were unchanged; its hash and the unchanged bindings source are recorded. Changed Python/CuTe/CUDA planner sources were compiled from this snapshot.

All completed validation commands exited zero. `matrix_01.exit=1` is an intentional preserved partial attempt, not a passing complete cohort. Final GPU memory/utilization checks showed all eight devices idle; the user-owned notebook was left running.

Only planner execution is timed. Expert/token payload, GEMM, optimizer and full-model training throughput are outside that timer. Default sampled_group reduced remote token destinations by 24.9794% and preserved all original per-layer owner hashes, but remains slower than the native planner.

Re-run independent readback from the repository root:

```bash
python experiments/configurable_planner_validation_20261105/audit_readback.py \
  --source-root . \
  --evidence-root experiments/configurable_planner_validation_20261105/evidence \
  --previous-benchmark experiments/native_sampled_validation_20261105/native8/evidence/planner_native8_attempt01.json \
  --output /tmp/configurable_readback_fresh.json
```
