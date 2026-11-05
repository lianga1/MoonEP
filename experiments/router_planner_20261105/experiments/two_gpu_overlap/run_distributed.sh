#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$task_root"
tag=$1
shift
case "$tag" in *[!a-zA-Z0-9_-]*|'') exit 2;; esac
out="$task_root/outputs/two_gpu_overlap"
mkdir -p "$out"
test ! -e "$out/$tag.exitcode"
trap 'rc=$?; printf "%s\n" "$rc" > "$out/$tag.exitcode"; date -u +%FT%TZ > "$out/$tag.finished"' EXIT
date -u +%FT%TZ > "$out/$tag.started"
printf '%s\n' "$@" > "$out/$tag.arguments"
# Use the activated environment, optionally selecting its bin directory.
if [[ -n "${MOONEP_PYTHON_BIN:-}" ]]; then export PATH="$MOONEP_PYTHON_BIN:$PATH"; fi
export PYTHONPATH="$task_root/inputs/MoonEP:$task_root/experiments/two_gpu_overlap/vendor/DeepEP_v1${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$task_root/build/torch_extensions"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1 MOONEP_MEM_HANDLE_TYPE=fd
torchrun --standalone --nproc_per_node=2 "$@" > "$out/$tag.log" 2>&1
