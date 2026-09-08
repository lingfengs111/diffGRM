#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 N_NEGATIVES" >&2
  exit 2
fi

n_negatives=$1
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_root="$repo/runs/sampled_catalog/video23_half_pairwise"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_half_2x2_d176_v1/pytorch_model.bin"
output_dir="$run_root/uniform_corrected_k${n_negatives}"
mkdir -p "$run_root"
cd "$repo"
test -s "$ar_ckpt"

# Match the successful half-size random four-head + pairwise + half-AR row.
# Only its training-time full-catalog CE is replaced by target + K uniform
# legal-item negatives with an importance-corrected denominator estimate.
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --sid-config experiments/capacity_fairness/video23_half_2x2_d176.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_half_2x2_d176_v1/pytorch_model.bin \
  --ar-checkpoint "$ar_ckpt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 35 \
  --training-objective catalog_plus_token \
  --sampled-catalog-negatives "$n_negatives" \
  --epochs 40 --patience 8 --min-epochs 10 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$output_dir" \
  >"$run_root/uniform_corrected_k${n_negatives}.log" 2>&1
