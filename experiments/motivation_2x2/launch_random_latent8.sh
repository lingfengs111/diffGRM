#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
run_id="beauty14_opq_random_latent8_ar_v1"
log_path="$repo_dir/runs/motivation_2x2/$run_id.log"

cd "$repo_dir"
env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/main.py" \
    --model=AR_GRM \
    --dataset=AmazonReviews2014 \
    --config="$repo_dir/experiments/motivation_2x2/common.yaml" \
    --config="$repo_dir/experiments/motivation_2x2/opq_random_latent8.yaml" \
    --config="$repo_dir/experiments/motivation_2x2/ar_latent.yaml" \
    --run_id="$run_id" \
    >"$log_path" 2>&1
