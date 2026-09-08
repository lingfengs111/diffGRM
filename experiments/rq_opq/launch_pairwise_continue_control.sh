#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
run_root="$repo_dir/runs/next_round_20260828"
while [[ ! -s "$run_root/video23_mips_pairwise_k32_overlap.json" ]]; do
    sleep 20
done

env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    /home/lingfengs111/miniconda3/envs/diffgrm/bin/python \
    "$repo_dir/scripts/train_parallel_opq_drafter.py" \
    --dataset AmazonReviews2023CleanGR \
    --common-config "$repo_dir/experiments/canonical_full/video23_cf_official.yaml" \
    --ar-config "$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml" \
    --diffusion-config "$repo_dir/experiments/canonical_full/diffusion_sequential.yaml" \
    --diffusion-checkpoint "$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin" \
    --ar-checkpoint "$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin" \
    --init-trained-checkpoint "$repo_dir/runs/parallel_drafter/video23_pairwise_diffenc_full_v1/best.pt" \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 32 \
    --epochs 2 --batch-size 256 --eval-batch-size 64 \
    --backbone-lr 0.0001 --selector-lr 0.001 \
    --token-loss-weight 0.1 --proposal-k 72 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --output-dir "$run_root/video23_pairwise_continue2_v1" \
    >"$run_root/video23_pairwise_continue2_v1.log" 2>&1

