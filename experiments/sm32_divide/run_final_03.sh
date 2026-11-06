#!/bin/bash
set -eu
task_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_sm32_divide_20261106
baseline_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_sm_budget_20261106
cd "$task_root"
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$task_root/build/extensions"
run() {
 name="$1";shift
 set +e
 "$@" > "$task_root/evidence/$name.log" 2>&1
 result=$?
 set -e
 echo "$result" > "$task_root/evidence/$name.exit"
 if [ "$result" != 0 ]; then exit "$result"; fi
}
run probe_selective03 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_sm_budget --fixture-root fixtures --sm-limit 32 --fuse --layers 0 --samples 5 --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output evidence/probe_selective03.json --profile-dir evidence/probe_selective03_traces
run config03 python -m pytest -q tests/test_planner_config.py
run components03 python -m pytest -q -s tests/test_sampled_sm_budget.py -k 'exchange or selector'
run tiles03 python -m pytest -q -s tests/test_sampled_exchange_tiles.py
run public03 python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_sm_budget.py -k 'not exchange and not selector'
run compatibility03 python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_candidate_limit.py -k 'not selector'
date -u > evidence/correctness_complete03
# Same full production boundary and benchmark source in old/new runtimes.
for variant in bounded fused; do
 flags=()
 if [ "$variant" = fused ]; then flags+=(--fuse); fi
 cd "$baseline_root"
 run "control_${variant}_before03" python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_sm_budget --fixture-root fixtures --sm-limit 32 "${flags[@]}" --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output "$task_root/evidence/control_${variant}_before03.json"
done
cd "$task_root"
for variant in bounded selective; do
 flags=()
 if [ "$variant" = selective ]; then flags+=(--fuse); fi
 run "final_${variant}03" python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_sm_budget --fixture-root fixtures --sm-limit 32 "${flags[@]}" --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output "evidence/final_${variant}03.json" --profile-dir "evidence/final_${variant}03_traces"
 run "profile_${variant}03" python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.profile_sampled_sm32 --fixture-root fixtures --sm-limit 32 "${flags[@]}" --layers 0,1,2,3,4,5,6,7 --output "evidence/profile_${variant}03.json"
done
for variant in bounded fused; do
 flags=()
 if [ "$variant" = fused ]; then flags+=(--fuse); fi
 cd "$baseline_root"
 run "control_${variant}_after03" python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_sm_budget --fixture-root fixtures --sm-limit 32 "${flags[@]}" --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output "$task_root/evidence/control_${variant}_after03.json"
done
date -u > "$task_root/evidence/final_complete03"
