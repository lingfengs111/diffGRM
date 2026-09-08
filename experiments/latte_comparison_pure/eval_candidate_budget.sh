#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU science23_opq4|video23_opq4 K}"
arm="${2:?usage: $0 GPU ARM K}"
proposal_k="${3:?usage: $0 GPU ARM K}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python

case "$arm" in
  science23_opq4|video23_opq4) ;;
  *) echo "unsupported arm: $arm" >&2; exit 2 ;;
esac
case "$proposal_k" in
  128|256) ;;
  *) echo "K must be 128 or 256" >&2; exit 2 ;;
esac

root="$repo/runs/latte_comparison_pure/$arm"
source_checkpoint="$root/onepass_pairwise_ar/best.pt"
ar_run="${arm}_pure_ar_l20_v1"
ar_checkpoint="$repo/saved/AmazonReviews2023CleanGR_${ar_run}/pytorch_model.bin"
sid_config="experiments/latte_comparison_pure/${arm}.yaml"
output="$root/candidate_budget/k${proposal_k}"
mkdir -p "$output"
cd "$repo"
test -s "$source_checkpoint"
test -s "$ar_checkpoint"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --sid-config "$sid_config" \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --ar-checkpoint "$ar_checkpoint" \
  --init-trained-checkpoint "$source_checkpoint" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
  --epochs 0 --batch-size 256 --eval-batch-size 32 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k "$proposal_k" --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$output" >"$root/candidate_budget/k${proposal_k}.log" 2>&1
