#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-3}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/sports14_transfer_l50"
ar_run=sports14_full_opq_cf_ar_l50_long_v1
ar_ckpt="$repo/saved/AmazonReviews2014CleanGR_${ar_run}/pytorch_model.bin"
stage1="$root/random_encoder4_pairwise_ar/stage1_12ep"
stage2="$root/random_encoder4_pairwise_ar/converged"
mkdir -p "$root/random_encoder4_pairwise_ar"
cd "$repo"

if [[ ! -s "$ar_ckpt" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2014CleanGR \
    --config=experiments/amazon14_domains/common_l50.yaml \
    --config=experiments/amazon14_domains/sports14_l50.yaml \
    --config=experiments/canonical_full/video23_ar_constrained.yaml \
    --max_history_len=50 --max_hist_len=50 \
    --run_id="$ar_run" >"$root/$ar_run.log" 2>&1
fi

test -s "$ar_ckpt"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2014CleanGR \
  --common-config experiments/amazon14_domains/common_l50.yaml \
  --sid-config experiments/amazon14_domains/sports14_l50.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --ar-checkpoint "$ar_ckpt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
  --epochs 12 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$stage1" >"$root/random_encoder4_pairwise_ar/stage1.log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2014CleanGR \
  --common-config experiments/amazon14_domains/common_l50.yaml \
  --sid-config experiments/amazon14_domains/sports14_l50.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --ar-checkpoint "$ar_ckpt" \
  --init-trained-checkpoint "$stage1/best.pt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
  --epochs 50 --patience 8 --min-epochs 6 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0001 --selector-lr 0.0005 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$stage2" >"$root/random_encoder4_pairwise_ar/converged.log" 2>&1

