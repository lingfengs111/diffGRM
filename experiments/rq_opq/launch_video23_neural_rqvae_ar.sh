#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
run_root="$repo_dir/runs/next_round_20260828"

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/main.py" \
    --model AR_GRM --dataset AmazonReviews2023CleanGR \
    --config "$repo_dir/experiments/canonical_full/video23_cf_official.yaml" \
    --config "$repo_dir/experiments/rq_opq/video23_neural_rqvae_text_cf.yaml" \
    --config "$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml" \
    --run_id=video23_full_neural_rqvae_text_cf_ar_v1 \
    >"$run_root/video23_full_neural_rqvae_text_cf_ar_v1.log" 2>&1
