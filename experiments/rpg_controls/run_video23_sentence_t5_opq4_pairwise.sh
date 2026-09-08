#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
output="$repo/runs/rpg_controls/video23_l20_sentence_t5_opq4/catalog_ce_pairwise_r32"
mkdir -p "$output"
cd "$repo"

# Strict third cell of the existing RPG-style control: the same collision-free
# Sentence-T5 OPQ4 catalog, random encoder-four-head backbone, optimization
# horizon, and catalog+token objective as catalog_ce_unary.  Only the low-rank
# all-pairs tuple selector is added.
exec env TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  "$python_bin" scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --sid-config experiments/amazon23_domains/video23.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_l20_v1/pytorch_model.bin \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random \
  --encoder-head-n-layer 4 \
  --conditioner diffusion_encoder \
  --variant pairwise \
  --pair-rank 32 \
  --training-objective catalog_plus_token \
  --token-loss-weight 0.1 \
  --epochs 30 \
  --patience 5 \
  --min-epochs 10 \
  --batch-size 256 \
  --eval-batch-size 128 \
  --backbone-lr 0.0001 \
  --selector-lr 0.001 \
  --weight-decay 0.0001 \
  --proposal-k 72 \
  --selection-metric ndcg10 \
  --skip-ar-verifier \
  --seed 2026 \
  --output-dir "$output"
