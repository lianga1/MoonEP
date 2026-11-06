#!/bin/bash
set -u
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$PWD/build/extensions"
mkdir -p evidence
run_check() {
  name="$1";shift
  "$@" > "evidence/${name}.log" 2>&1
  result=$?
  echo "$result" > "evidence/${name}.exit"
  if [ "$result" != 0 ]; then exit "$result"; fi
}
run_check probe_fused32_01 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_sm_budget --fixture-root fixtures --candidate-limit 128 --rounds 4 --sm-limit 32 --fuse --layers 0 --samples 3 --replays 2 --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --profile-dir evidence/probe_fused32_traces_01 --output evidence/probe_fused32_01.json
run_check components_01 python -m pytest -q -s tests/test_sampled_sm_budget.py -k 'exchange or selector'
run_check public_01 python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_sm_budget.py -k 'not exchange and not selector'
run_check compatibility_01 python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_candidate_limit.py -k 'not selector'
date -u > evidence/correctness_complete_01
