#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU SEED" >&2
  exit 2
fi

gpu="$1"
seed="$2"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/paper_anchor_20260903/beauty14_encoder4_pairwise_seed${seed}"
stage1="$root/stage1_10ep"
stage2="$root/converged"
mkdir -p "$root"
cd "$repo"

# Stage 1 mirrors the from-scratch Video23 encoder-four-head experiment.
CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2014CleanGR \
  --common-config experiments/canonical_full/beauty14_full.yaml \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --ar-checkpoint saved/AmazonReviews2014CleanGR_beauty14_full_opq_ar_finetune_control_v1/pytorch_model.bin \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 4 \
  --variant pairwise \
  --conditioner diffusion_encoder \
  --pair-rank 51 \
  --epochs 10 \
  --batch-size 256 \
  --eval-batch-size 64 \
  --backbone-lr 0.0003 \
  --selector-lr 0.001 \
  --weight-decay 0.0001 \
  --token-loss-weight 0.1 \
  --proposal-k 72 \
  --seed "$seed" \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$stage1" \
  >"$root/stage1.log" 2>&1

# Stage 2 matches the long-horizon continuation that produced the current
# strongest full-data Video23 result. Validation, not the test set, chooses
# both the checkpoint and fusion weight.
CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2014CleanGR \
  --common-config experiments/canonical_full/beauty14_full.yaml \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --ar-checkpoint saved/AmazonReviews2014CleanGR_beauty14_full_opq_ar_finetune_control_v1/pytorch_model.bin \
  --init-trained-checkpoint "$stage1/best.pt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 4 \
  --variant pairwise \
  --conditioner diffusion_encoder \
  --pair-rank 51 \
  --epochs 30 \
  --patience 5 \
  --min-epochs 5 \
  --batch-size 256 \
  --eval-batch-size 64 \
  --backbone-lr 0.0001 \
  --selector-lr 0.0005 \
  --weight-decay 0.0001 \
  --token-loss-weight 0.1 \
  --proposal-k 72 \
  --seed "$seed" \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$stage2" \
  >"$root/converged.log" 2>&1

