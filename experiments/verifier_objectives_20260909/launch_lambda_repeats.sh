#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
runner="$repo/experiments/verifier_objectives_20260909/run_lambda_repeat.sh"
root="$repo/runs/verifier_objectives_20260909/lambda_dpo_repeats_e12"
seeds=(2026 2027 2028 2029)
pids=()

mkdir -p "$root"
for gpu in 0 1 2 3; do
  bash "$runner" "$gpu" "${seeds[$gpu]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  echo "one or more Lambda-DPO repeats failed" >&2
  exit 1
fi
date --iso-8601=seconds >"$root/SUITE_COMPLETE"
