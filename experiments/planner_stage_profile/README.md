# Eight-rank planner profiling, 2026-11-05

See [the measured analysis](../../docs/sampled_stage_profile.md). All performance
runs used eight physical H200 GPUs on the authorized `omni-dev` instance.
Original router inputs are not included here. Reproduction of the trace benchmark
requires the existing eight `layer_00.npz` … `layer_07.npz` fixtures; public
integration tests do not require private router data.

## Evidence and source identity

- `full_baseline_01.json` and `full_optimized_02.json`: all eight layers, unchanged
  uninstrumented full-planner timing protocol and physical-plan audits.
- `baseline_01.json` and `optimized_02.json`: layers 0/7, 36 stage events plus
  exact output/history/package/provenance comparison to production execution.
- `*_traces/`: original losslessly compressed all-rank eager mapping and warmed
  Graph captures. Eager timings are not used as Graph timings.
- `*_trace_analysis_*.json.gz`: conservative offline attribution, full event
  indices, kernel sum/union/span and mapping coverage. These are diagnostic
  captures, not the uninstrumented planner latency.
- `selector_exact_02.json`: 17 CPU-oracle comparisons, including full raw history,
  changing inputs on one Graph, invalid-copy recovery and generic fallback.
- `integration_02.log`: 6 tests passed on each of 8 physical ranks.
- `host_config_02.log`: 13 tests passed. Every corresponding `.exit` is zero.
- `optimized_01` retains the slower all-table-rebuild attempt. `optimized_03`
  retains the 512-thread trial; its small additional gain did not justify
  changing the default. Exact candidate source snapshots are in `candidates/`.

Production source SHA256 is recorded in each JSON. The baseline is commit
`81a5a31459d940f41d60ec384f3c3fe1920d339a`; the final CUDA source SHA256 is
`c778fcfd411cb2a82d7238c6075f1cf59ba7e0f017184a9270747158c9471c47`.
The accepted configuration keeps 256 threads. A shared-storage limit of 48 KiB
selects the unchanged generic kernel for larger configurations.

## Reproduce

Run from a built checkout, with the original fixture directory linked as
`fixtures`. Use fresh output paths; never overwrite earlier attempts. Keep GPU
benchmarks and correctness jobs sequential.

```bash
export PYTHONPATH=.
export MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$PWD/build/extensions"

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tests.profile_sampled_stages --fixture-root fixtures \
  --output /tmp/new_stages.json --profile-dir /tmp/new_traces
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tests.benchmark_sampled_group --fixture-root fixtures --output /tmp/new_full.json
python -m tests.validate_sampled_residency_table --fixture-root fixtures \
  --output /tmp/new_exact.json --cases trace,b44,fallback,small,highg,k16
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m pytest -q -s tests/test_sampled_group.py tests/test_sampled_weights.py
python -m pytest -q tests/test_planner_config.py
```

The archived `run_validation_02.sh` is the actual sequential validation runner;
it records each command's exit code separately. `validation_02.finished` alone
does not imply every test passed. Inspect all exit codes and JSON failures.

Offline, standard-library-only trace analysis:

```bash
python experiments/planner_stage_profile/analyze_traces.py \
  --trace-dir /tmp/new_traces --output /tmp/new_analysis.json --layers 0,7
```

`build_report.py --help` describes the comparison report. It accepts both JSON
and gzip-compressed analysis files, validates production source/shape identity,
and keeps full eight-layer latency separate from two-layer diagnostic timing.
