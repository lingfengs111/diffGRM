#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/rpg_controls/video23_sentence_t5_opq4"
mkdir -p "$root"
cd "$repo"

common=(
  --dataset AmazonReviews2023CleanGR
  --common-config experiments/canonical_full/video23_cf_official.yaml
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin
  --backbone-architecture encoder_four_head
  --backbone-initialization random
  --encoder-head-n-layer 4
  --variant unary
  --conditioner diffusion_encoder
  --skip-ar-verifier
  --epochs 30
  --patience 5
  --min-epochs 10
  --batch-size 256
  --eval-batch-size 128
  --weight-decay 0.0001
  --proposal-k 72
  --seed 2026
  --selection-metric ndcg10
)

# Controlled RPG row: identical data, collision-free OPQ4 catalog and
# encoder-four-head backbone, but only independent multi-token prediction.
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  "${common[@]}" \
  --training-objective mtp_only \
  --mtp-temperature 0.07 \
  --backbone-lr 0.0003 \
  --output-dir "$root/rpg_mtp_unary" \
  >"$root/rpg_mtp_unary.log" 2>&1

# Matched objective control: same initialization family and architecture, but
# optimize the legal item catalog directly (plus the canonical token anchor).
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  "${common[@]}" \
  --training-objective catalog_plus_token \
  --token-loss-weight 0.1 \
  --backbone-lr 0.0001 \
  --output-dir "$root/catalog_ce_unary" \
  >"$root/catalog_ce_unary.log" 2>&1
