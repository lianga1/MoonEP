# Candidate-limit 128 evidence

This directory preserves the experiment used to select **128 candidates, four
exchange rounds, eight physical and logical GPUs**. It is historical evidence
from the active-cache wrapper around production baseline
`b1518bf9166880a22bb45937dc79458ea2e5102b`; fresh integrated-runtime checks are
preserved separately in [integration/](integration/README.md) and reported in
[the delivery document](../../docs/sampled_candidate_limit.md). The historical
`manifest.json` is unchanged by the new integration evidence.

The complete original four-round cohort is retained, including limits
0/64/128/256/512, all eight layers, ten samples and five graph replays per
sample, all rank timings, shape/environment metadata, source hashes and audit
results. This is not the separate physical-compute-scaling CUDA port.

Files:

- `full_active_rounds4_01.json.gz`: original JSON bytes in deterministic gzip.
  `manifest.json` records both compressed and original hashes.
- `full_active_rounds4_01.log` and `.exit`: original completion evidence.
- `cpu_summary_02.json`: independent full-trace simulation summary, including
  the native reference count and cap tradeoffs.
- `cpu_array_hashes.json`: CPU support, final owners, selection history and
  retained-candidate array hashes used by the GPU benchmark's checks.
- `historical_audit_excerpt.json`: four-round cohort from the independent
  audit, with the complete original audit's hash. References to other cohorts
  and CPU archives describe that historical audit; those other raw cohorts and
  large arrays are not part of this compact delivery.
- `summarize.py`: verifies the packaged file hashes and recomputes the selected
  comparison from the raw per-rank timing samples.

Run without CUDA or third-party Python packages:

```bash
python experiments/candidate_limit128/summarize.py
```

For the fresh production-interface run:

```bash
python experiments/candidate_limit128/integration/summarize.py
```

Traffic counts are deduplicated `(source token, remote destination rank)`
visits. Planner timing includes live native planning, sample/count collectives,
candidate pruning, expert selection, four exchange rounds, address generation,
status checks and metadata publication. It excludes input histogram creation,
JIT, allocations, audits, expert/token/gradient payload transfers, GEMM and
optimizer execution. Do not convert these results into a training-step speedup.
