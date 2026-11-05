#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-single}"
shift || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-4}"
suffix=""
for opt in "$@"; do
  if [[ "$opt" == "--graph" ]]; then suffix="_graph"; fi
done
case "$mode" in
  single) python benchmark.py --mode single --output "results/single${suffix}.json" "$@" ;;
  local|coordinated) torchrun --standalone --nproc_per_node=8 benchmark.py --mode "$mode" --output "results/${mode}${suffix}.json" "$@" ;;
  *) echo "usage: run.sh single|local|coordinated [benchmark options]" >&2; exit 2 ;;
esac
