#!/usr/bin/env bash
set -uo pipefail
cd /inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_configurable_20261105
export PYTHONPATH="$PWD"
export MOONEP_MEM_HANDLE_TYPE=fd
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4
export OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$PWD/build/extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR" evidence
attempt="$1"
log="evidence/${attempt}.log"
if [[ -e "$log" || -e "evidence/${attempt}.exit" ]]; then
  exit 91
fi
case "$attempt" in
  host_01) python -m pytest -q tests/test_planner_config.py > "$log" 2>&1 ;;
  matrix_*) python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -x -q -s tests/test_sampled_configurable.py > "$log" 2>&1 ;;
  default_*) python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -x -q -s tests/test_sampled_group.py tests/test_sampled_weights.py > "$log" 2>&1 ;;
  replay_*) python -m torch.distributed.run --standalone --nproc_per_node=2 tests/sampled_components.py --fixture-root fixtures --output "evidence/${attempt}.json" > "$log" 2>&1 ;;
  benchmark_*) python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_group --fixture-root fixtures --output "evidence/${attempt}.json" > "$log" 2>&1 ;;
  *) exit 92 ;;
esac
result=$?
printf '%s\n' "$result" > "evidence/${attempt}.exit"
exit "$result"
