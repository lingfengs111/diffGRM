#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_root="$repo/runs/capacity_fairness/video23_continuations_20260831"
mkdir -p "$run_root"
cd "$repo"

# Continue the six-epoch shared-history experiment from its actual best point.
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_shared_encoder_ar_verifier.py \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --drafter-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin \
  --init-shared-checkpoint runs/parameter_fairness/video23_full_20260830/shared_encoder_ar/best.pt \
  --pair-rank 51 --proposal-k 72 --candidate-score-chunk-size 16 \
  --epochs 12 --patience 4 --batch-size 256 --eval-batch-size 64 \
  --learning-rate 0.00005 --weight-decay 0.0001 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$run_root/shared_encoder_ar_continue" \
  >"$run_root/shared_encoder_ar_continue.log" 2>&1

# The strongest one-pass drafter also ended at its epoch cap.  Continue it
# gently to separate an equal-budget ablation from an approximate ceiling.
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin \
  --init-trained-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --backbone-architecture masked_decoder \
  --backbone-initialization diffusion_pretrained \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
  --epochs 30 --patience 5 --min-epochs 5 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0001 --selector-lr 0.0005 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$run_root/one_pass_pairwise_continue" \
  >"$run_root/one_pass_pairwise_continue.log" 2>&1
