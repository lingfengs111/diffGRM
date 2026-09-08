#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
checkpoint="${1:-$repo_dir/runs/next_round_20260828/video23_pairwise_rank51_continue2_v1/best.pt}"
output_dir="${2:-$repo_dir/runs/set_diffusion/round1_video23/old_model_two_pass_preserve_b16}"
gpu="${CUDA_VISIBLE_DEVICES:-2}"

exec env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/train_parallel_opq_drafter.py" \
    --dataset AmazonReviews2023CleanGR \
    --common-config "$repo_dir/experiments/canonical_full/video23_cf_official.yaml" \
    --ar-config "$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml" \
    --diffusion-config "$repo_dir/experiments/canonical_full/diffusion_sequential.yaml" \
    --diffusion-checkpoint "$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin" \
    --ar-checkpoint "$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin" \
    --init-trained-checkpoint "$checkpoint" \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
    --epochs 0 --batch-size 128 --eval-batch-size 16 \
    --proposal-k 72 --two-pass-branches 16 --two-pass-branch-chunk 16 \
    --two-pass-first-weights 0,0.25,0.5,0.75,1 \
    --two-pass-preserve-first --skip-ar-verifier \
    --output-dir "$output_dir"
