#!/bin/bash
set -eu
source_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_sm32_divide_20261106
task_root=/inspire/hdd/project/embodied-multimodality/liangzuning-253108090053/moonep_sm128_validation_20261106
mkdir -p "$task_root/evidence"
cd "$source_root"
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$source_root/build/extensions"
run() {
 name="$1";shift
 set +e
 "$@" > "$task_root/evidence/$name.log" 2>&1
 result=$?
 set -e
 echo "$result" > "$task_root/evidence/$name.exit"
 if [ "$result" != 0 ]; then exit "$result"; fi
}
nvidia-smi > "$task_root/evidence/environment_before.txt"
nvidia-smi topo -m > "$task_root/evidence/topology.txt"
python -c 'import hashlib,json; from pathlib import Path; m=json.loads(Path("candidate_manifest_03.json").read_text()); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in m["files"].items()); print("runtime and harness exactly match frozen payload03 / commit55564bc")' > "$task_root/evidence/source_check.txt"
benchmark() {
 name="$1";shift
 run "$name" python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_sm_budget \
   --fixture-root fixtures --candidate-limit 128 --rounds 4 --samples 10 --replays 5 \
   --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json \
   --output "$task_root/evidence/$name.json" "$@"
}
benchmark selective32_before --sm-limit 32 --fuse
benchmark bounded128 --sm-limit 128 --profile-dir "$task_root/evidence/bounded128_traces"
benchmark selective128 --sm-limit 128 --fuse --profile-dir "$task_root/evidence/selective128_traces"
run profile128 python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.profile_sampled_sm32 \
  --fixture-root fixtures --sm-limit 128 --fuse --layers 0,1,2,3,4,5,6,7 \
  --samples 10 --replays 5 --output "$task_root/evidence/profile128.json"
benchmark selective32_after --sm-limit 32 --fuse
nvidia-smi > "$task_root/evidence/environment_after.txt"
date -u > "$task_root/evidence/complete"
