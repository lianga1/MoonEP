#!/bin/bash
set -eu
task_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_sm_budget_20261106
baseline_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_cap128_delivery_20261106
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$task_root/build/extensions"
test -f "$task_root/evidence/correctness_complete_03"
cp "$task_root/tests/benchmark_sampled_sm_budget.py" "$baseline_root/tests/benchmark_sampled_sm_budget.py"
cp "$task_root/tests/sampled_sm_trace.py" "$baseline_root/tests/sampled_sm_trace.py"
run_bench() {
  name="$1";repo="$2";shift 2
  cd "$repo"
  set +e
  python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_sm_budget \
    --fixture-root fixtures --candidate-limit 128 --rounds 4 \
    --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json \
    --output "$task_root/evidence/${name}.json" "$@" > "$task_root/evidence/${name}.log" 2>&1
  result=$?
  set -e
  echo "$result" > "$task_root/evidence/${name}.exit"
  if [ "$result" != 0 ]; then exit "$result"; fi
}
run_bench baseline_before_03 "$baseline_root" --profile-dir "$task_root/evidence/baseline_before_traces_03"
run_bench bounded32_03 "$task_root" --sm-limit 32 --profile-dir "$task_root/evidence/bounded32_traces_03"
for limit in 32 48 64 96 128; do
  run_bench "fused${limit}_03" "$task_root" --sm-limit "$limit" --fuse --profile-dir "$task_root/evidence/fused${limit}_traces_03"
done
run_bench baseline_after_03 "$baseline_root"
date -u > "$task_root/evidence/matrix_complete_03"
