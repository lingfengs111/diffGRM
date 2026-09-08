#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_root="$repo/runs/capacity_fairness/video23_depth_half_d256"
ar_id=video23_full_opq_cf_ar_depth_half_1x1_d256_v1
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_id}/pytorch_model.bin"
mkdir -p "$run_root"
cd "$repo"

# Keep the canonical d=256 width but halve the AR transformer's depth from
# 2 encoder + 2 decoder blocks to 1 + 1.  This is trained from scratch on the
# same collision-free OPQ4 representation and full Video23 split.
env TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=AR_GRM \
  --dataset=AmazonReviews2023CleanGR \
  --config=experiments/canonical_full/video23_cf_official.yaml \
  --config=experiments/canonical_full/video23_ar_constrained.yaml \
  --config=experiments/capacity_fairness/video23_depth_half_1x1_d256.yaml \
  --run_id="$ar_id" \
  >"$run_root/$ar_id.log" 2>&1

test -s "$ar_ckpt"

# The drafter remains one-pass and random-initialized; only its history
# encoder is reduced from four full-width blocks to two.  K=72 is retained in
# this capacity experiment so its only changed variable is model depth.
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --sid-config experiments/capacity_fairness/video23_depth_half_1x1_d256.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --ar-checkpoint "$ar_ckpt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 2 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 35 \
  --training-objective catalog_plus_token \
  --epochs 40 --patience 8 --min-epochs 10 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$run_root/random_encoder2_pairwise_depth_ar" \
  >"$run_root/random_encoder2_pairwise_depth_ar.log" 2>&1
