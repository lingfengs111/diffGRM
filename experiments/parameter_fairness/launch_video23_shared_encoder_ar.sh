#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
out="$repo/runs/parameter_fairness/video23_full_20260830/shared_encoder_ar"
mkdir -p "$out"
cd "$repo"

exec /home/lingfengs111/miniconda3/envs/diffgrm/bin/python \
  scripts/train_shared_encoder_ar_verifier.py \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --drafter-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin \
  --pair-rank 51 --proposal-k 72 --candidate-score-chunk-size 16 \
  --epochs 6 --patience 2 --batch-size 256 --eval-batch-size 64 \
  --learning-rate 0.0001 --weight-decay 0.0001 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$out"
