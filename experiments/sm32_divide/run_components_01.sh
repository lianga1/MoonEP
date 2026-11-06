#!/bin/bash
set -eu
task_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_sm32_divide_20261106
cd "$task_root"
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$task_root/build/extensions"
run() {
 name="$1";shift
 set +e
 "$@" > "evidence/$name.log" 2>&1
 result=$?
 set -e
 echo "$result" > "evidence/$name.exit"
 if [ "$result" != 0 ]; then exit "$result"; fi
}
run compact_probe01 python -m tests.benchmark_residency_compact --fixture-root fixtures --output evidence/compact_probe01.json --layers 0 --samples 5 --replays 20
run exchange_tiles01 python -m tests.benchmark_sampled_exchange_tiles --fixture-root fixtures --output evidence/exchange_tiles01.json --samples 10 --replays 20
run compact_all01 python -m tests.benchmark_residency_compact --fixture-root fixtures --output evidence/compact_all01.json
run tile_correctness01 python -m pytest -q -s tests/test_sampled_exchange_tiles.py
run components_correctness01 python -m pytest -q -s tests/test_sampled_sm_budget.py -k 'exchange or selector'
date -u > evidence/components_complete_01
