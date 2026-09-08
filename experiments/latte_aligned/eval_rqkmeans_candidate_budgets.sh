#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU science23|video23}"
domain="${2:?usage: $0 GPU science23|video23}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python

case "$domain" in
  science23) ar_run=science23_latte_rqkmeans3_latent8_ar_l20_v1 ;;
  video23) ar_run=video23_latte_rqkmeans3_latent8_ar_l20_v1 ;;
  *) echo "unknown domain: $domain" >&2; exit 2 ;;
esac

root="$repo/runs/latte_aligned/$domain"
source_ckpoint="$root/onepass_pairwise_ar/best.pt"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_run}/pytorch_model.bin"
sid_config="experiments/latte_aligned/${domain}_rqkmeans3_latent8.yaml"
mkdir -p "$root/candidate_budget"
cd "$repo"
test -s "$source_ckpoint"
test -s "$ar_ckpt"

for proposal_k in 128 256; do
  output="$root/candidate_budget/k${proposal_k}"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config "$sid_config" \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --ar-checkpoint "$ar_ckpt" \
    --init-trained-checkpoint "$source_ckpoint" \
    --backbone-architecture encoder_four_head \
    --backbone-initialization random --encoder-head-n-layer 4 \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
    --epochs 0 --batch-size 256 --eval-batch-size 16 \
    --backbone-lr 0.0003 --selector-lr 0.001 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k "$proposal_k" --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --output-dir "$output" >"$root/candidate_budget/k${proposal_k}.log" 2>&1
done
