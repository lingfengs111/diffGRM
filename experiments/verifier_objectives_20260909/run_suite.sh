#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_suite.sh smoke|full}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
runner="$repo/experiments/verifier_objectives_20260909/run_arm.sh"
root="$repo/runs/verifier_objectives_20260909/$stage"
arms=(apao_all apao_support ar_hard_support lambda_dpo_support)
pids=()

mkdir -p "$root"
for gpu in 0 1 2 3; do
  arm="${arms[$gpu]}"
  bash "$runner" "$gpu" "$arm" "$stage" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  echo "one or more $stage arms failed" >&2
  exit 1
fi
date --iso-8601=seconds >"$root/SUITE_COMPLETE"
