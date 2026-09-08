#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/long_horizon_20260831"
mkdir -p "$root"
cd "$repo"

run_control() {
  local name="$1"
  local init_checkpoint="$2"
  local architecture="$3"
  shift 3
  env TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/canonical_full/video23_cf_official.yaml \
    --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
    --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
    --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
    --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin \
    --init-trained-checkpoint "$init_checkpoint" \
    --backbone-architecture "$architecture" \
    --backbone-initialization random \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
    --epochs 30 --patience 5 --min-epochs 5 \
    --batch-size 256 --eval-batch-size 64 \
    --backbone-lr 0.0001 --selector-lr 0.0005 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k 72 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    "$@" --output-dir "$root/$name" >"$root/$name.log" 2>&1
}

run_control masked_random_pairwise_converged \
  runs/diffusion_necessity/video23_full_20260829/masked_random_pairwise_r51/best.pt \
  masked_decoder

run_control encoder4_four_head_pairwise_converged \
  runs/diffusion_necessity/video23_full_20260829/encoder4_four_head_pairwise_r51/best.pt \
  encoder_four_head --encoder-head-n-layer 4

