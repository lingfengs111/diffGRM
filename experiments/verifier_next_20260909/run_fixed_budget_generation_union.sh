#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-2}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/verifier_next_20260909/fixed_budget_generation_union/video23_d56_a16_k72"
mkdir -p "$root"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/evaluate_generation_augmented_fusion.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --sid-config experiments/latte_comparison_pure/video23_opq4.yaml \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --drafter-checkpoint runs/latte_comparison_pure/video23_opq4/onepass_pairwise_ar/best.pt \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_opq4_pure_ar_l20_v1/pytorch_model.bin \
  --candidate-k 72 --draft-k 56 --ar-k 16 --ar-search-beam 16 \
  --eval-batch-size 32 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output "$root/result.json" >"$root/eval.log" 2>&1
test -s "$root/result.json"
date --iso-8601=seconds >"$root/COMPLETE"
