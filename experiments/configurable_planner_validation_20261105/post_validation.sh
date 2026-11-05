#!/usr/bin/env bash
set -uo pipefail
cd /inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_configurable_20261105
export PYTHONPATH="$PWD" MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$PWD/build/extensions"
while [[ ! -f evidence/matrix_01.exit ]]; do sleep 10; done
[[ "$(cat evidence/matrix_01.exit)" == 0 ]] || exit 93
bash run_validation.sh default_01 || exit "$?"
bash run_validation.sh replay_01 || exit "$?"
python -m pytest -x -q -s tests/test_sampled_residency_configurable.py > evidence/residency_01.log 2>&1
result=$?
printf '%s\n' "$result" > evidence/residency_01.exit
[[ "$result" == 0 ]] || exit "$result"
python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -x -q -s tests/test_sampled_configurable_weighted.py > evidence/weighted_01.log 2>&1
result=$?
printf '%s\n' "$result" > evidence/weighted_01.exit
[[ "$result" == 0 ]] || exit "$result"
bash run_validation.sh benchmark_01
