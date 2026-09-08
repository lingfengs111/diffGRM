#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
run_id="beauty14_full_opq_pit_alias2_ar_v1"
output_dir="$repo_root/runs/canonical_full/pit_lite"
mkdir -p "$output_dir"

CUDA_VISIBLE_DEVICES="${1:-0}" "$python_bin" "$repo_root/main.py" \
    --model=AR_GRM \
    --dataset=AmazonReviews2014CleanGR \
    --config="$repo_root/experiments/canonical_full/beauty14_full.yaml" \
    --config="$repo_root/experiments/canonical_full/ar_constrained.yaml" \
    --config="$repo_root/experiments/pit_lite/beauty14_alias2_ar.yaml" \
    --run_id="$run_id" \
    >"$output_dir/$run_id.log" 2>&1 &

pid=$!
echo "$pid" >"$output_dir/$run_id.pid"
echo "launched gpu=${1:-0} pid=$pid run_id=$run_id"
