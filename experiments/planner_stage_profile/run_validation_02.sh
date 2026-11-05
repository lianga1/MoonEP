#!/usr/bin/env bash
set -u
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4 OMP_NUM_THREADS=1 TORCH_EXTENSIONS_DIR="$PWD/build/extensions"
run_case() {
  local label="$1"
  shift
  "$@" > "evidence/${label}.log" 2>&1
  local result=$?
  echo "$result" > "evidence/${label}.exit"
  return 0
}
run_case full_optimized_02 python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tests.benchmark_sampled_group --fixture-root fixtures --output evidence/full_optimized_02.json
run_case selector_exact_02 python -m tests.validate_sampled_residency_table \
  --fixture-root fixtures --output evidence/selector_exact_02.json --cases trace,b44,fallback,small,highg,k16
run_case integration_02 python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m pytest -q -s tests/test_sampled_group.py tests/test_sampled_weights.py
run_case host_config_02 python -m pytest -q tests/test_planner_config.py
date -u > evidence/validation_02.finished
