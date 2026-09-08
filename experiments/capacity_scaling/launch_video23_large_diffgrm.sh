#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/capacity_scaling/video23_large_20260830"
common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
large="$repo_dir/experiments/capacity_scaling/video23_large_4x4_d384_ffn1536.yaml"
run_id="video23_full_opq_cf_diff_guided_large_4x4_v1"
mkdir -p "$output_dir"

echo "$(date -Is) launching gpu=0 model=DIFF_GRM run_id=$run_id" \
    | tee -a "$output_dir/launcher.log"
env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/main.py" \
    --model=DIFF_GRM \
    --dataset=AmazonReviews2023CleanGR \
    --config="$common" \
    --config="$large" \
    --run_id="$run_id" \
    >"$output_dir/$run_id.log" 2>&1

echo "$(date -Is) completed run_id=$run_id" \
    | tee -a "$output_dir/launcher.log"
