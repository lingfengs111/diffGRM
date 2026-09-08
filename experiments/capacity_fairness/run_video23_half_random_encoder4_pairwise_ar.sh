#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_root="$repo/runs/capacity_fairness/video23_half_2x2_d176"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_half_2x2_d176_v1/pytorch_model.bin"
mkdir -p "$run_root"
cd "$repo"

# This capacity control is intentionally independent of DiffGRM denoising
# pretraining.  It asks whether a random, half-width four-head semantic drafter
# plus a half-width AR verifier retains the gain of the full two-model system.
test -s "$ar_ckpt"
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --sid-config experiments/capacity_fairness/video23_half_2x2_d176.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_half_2x2_d176_v1/pytorch_model.bin \
  --ar-checkpoint "$ar_ckpt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 35 \
  --epochs 40 --patience 8 --min-epochs 10 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$run_root/random_encoder4_pairwise_half_ar" \
  >"$run_root/random_encoder4_pairwise_half_ar.log" 2>&1
