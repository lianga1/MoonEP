#!/bin/bash
set -eu
task_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_sm32_divide_20261106
cd "$task_root"
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$task_root/build/extensions"
run() {
 name="$1";shift
 set +e
 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.profile_sampled_sm32 --fixture-root fixtures --output "evidence/$name.json" "$@" > "evidence/$name.log" 2>&1
 result=$?
 set -e
 echo "$result" > "evidence/$name.exit"
 if [ "$result" != 0 ]; then exit "$result"; fi
}
run profile_probe01 --layers 0 --samples 3 --stage-samples 3 --trace-replays 3
run baseline_bounded32_01 --layers 0,1,2,3,4,5,6,7
run baseline_fused32_01 --fuse --layers 0,1,2,3,4,5,6,7
date -u > evidence/baseline_profile_complete_01
