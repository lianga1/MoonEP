# Live eight-rank sampled_group integration evidence

Base: `6fb5f9df437fbe2d92d7279171254adb252a5b7a`; the accompanying commit adds the native selectable backend and tests. [Usage and validation summary](../../docs/sampled_group_validation.md).

- `native8/`: actual eight-H200 omni-dev build and public runtime tests, failed gradient compilation attempt, successful corrected weight/gradient test, full eight-rank planner benchmark, and final remote source hashes.
- `components2/`: separate two-H200 component regression using the original eight source shards; this is not an eight-physical-rank measurement.
- `independent_readback.json`: re-computation of timing statistics, fixtures, counts and source hashes from retained evidence.
- `publication_manifest.json`: source files copied into the fork and their hashes. Pre-existing unrelated edits were preserved.
- `residency_static_audit.json`: initialization/sampling, mandatory support and B44 history-bound CPU/static checks; separate from the GPU tests.

Successful exit codes: native public runtime, full planner benchmark, corrected weight/gradient test, and two-GPU components are all 0. `native8_weights_attempt02.exitcode=1` is deliberately retained: the original gradient reducer called removed CuTe `make_fragment`; replacing it with `make_rmem_tensor` passed the real eight-rank retry. All raw logs retain their original paths and software diagnostics.

The planner benchmark predates that gradient-only compatibility fix. Its source snapshot differs only at `moonep/grad_reduce.py`; the benchmark does not invoke gradient reduction. Final remote hashes and the successful gradient test establish the corrected source separately. CUDA launch code used for planner timing is unchanged.

The full planner is slower than native MoonEP: 1.286090 ms versus 0.068024 ms per layer on this trace, averaged over layer-wise worst-rank medians. The 24.9794% drop in remote token destinations is not a claim of end-to-end training speedup. Expert/GEMM/optimizer execution is not included in the planner timer; the expert transport test checks numerical transport and reduction rather than training throughput.

`independent_readback_audit_attempt01.json` retains a corrected audit-helper assumption about variable-length accepted histories. It was not a GPU execution failure; the final independent readback checks each layer against its own manifest.
