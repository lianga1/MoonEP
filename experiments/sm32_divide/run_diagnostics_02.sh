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
run compact_all02 python -m tests.benchmark_residency_compact --fixture-root fixtures --output evidence/compact_all02.json
run frequency_shards02 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_frequency_sharding --fixture-root fixtures --output evidence/frequency_shards02.json
run peer_grids02 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_peer_grids --output evidence/peer_grids02.json --samples 20
run full_unfused_probe02 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_divide --fixture-root fixtures --sm-limit 32 --layers 0 --samples 5 --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output evidence/full_unfused_probe02.json --profile-dir evidence/full_unfused_probe02_traces
run full_fused_probe02 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_divide --fixture-root fixtures --sm-limit 32 --fuse --layers 0 --samples 5 --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output evidence/full_fused_probe02.json --profile-dir evidence/full_fused_probe02_traces
run full_selective_probe02 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_divide --fixture-root fixtures --sm-limit 32 --fuse --proposal-fuse 0 --layers 0 --samples 5 --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output evidence/full_selective_probe02.json --profile-dir evidence/full_selective_probe02_traces
date -u > evidence/diagnostics_complete_02
