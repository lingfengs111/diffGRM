#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/toys14_transfer/random_encoder4_pairwise_ar"
ar_ckpt="$repo/saved/AmazonReviews2014CleanGR_toys14_full_opq_cf_ar_l20_long_v1/pytorch_model.bin"
stage1="$root/stage1_12ep"
stage2="$root/converged"
mkdir -p "$root"
test -s "$ar_ckpt"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2014CleanGR \
  --common-config experiments/amazon14_domains/common_l20.yaml \
  --sid-config experiments/amazon14_domains/toys14.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --ar-checkpoint "$ar_ckpt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
  --epochs 12 --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$stage1" >"$root/stage1.log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2014CleanGR \
  --common-config experiments/amazon14_domains/common_l20.yaml \
  --sid-config experiments/amazon14_domains/toys14.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --ar-checkpoint "$ar_ckpt" --init-trained-checkpoint "$stage1/best.pt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
  --epochs 60 --patience 10 --min-epochs 8 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0001 --selector-lr 0.0005 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$stage2" >"$root/converged.log" 2>&1
