#!/bin/bash
# Portable equivalent of the preserved original_validation.sh.gz commands.
set -eu
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 /absolute/router-fixture-directory /new/output-directory" >&2
  exit 2
fi
fixture_root="$(cd "$1" && pwd)"
mkdir "$2"
output_dir="$(cd "$2" && pwd)"
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$repo_root"
export PYTHONPATH=. MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$PWD/build/extensions"
run_check() {
  name="$1"
  shift
  set +e
  "$@" > "$output_dir/${name}.log" 2>&1
  result=$?
  set -e
  echo "$result" > "$output_dir/${name}.exit"
  if [ "$result" != 0 ]; then exit "$result"; fi
}
run_check host_config_01 python -m pytest -q tests/test_planner_config.py
run_check selector_01 python -m pytest -q -s tests/test_sampled_candidate_limit.py -k selector
run_check cap128_public_01 python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_candidate_limit.py -k 'not selector'
run_check compatibility_public_01 python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_group.py
for cap in 0 128; do
  run_check "benchmark_cap${cap}_01" python -m torch.distributed.run --standalone --nproc_per_node=8 -m tests.benchmark_sampled_candidate_limit --fixture-root "$fixture_root" --candidate-limit "$cap" --rounds 4 --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json --output "$output_dir/benchmark_cap${cap}_01.json"
done
date -u > "$output_dir/complete_01"
