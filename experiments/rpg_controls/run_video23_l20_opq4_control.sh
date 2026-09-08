#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
mode="${1:?usage: run_video23_l20_opq4_control.sh mtp|unary}"
root="$repo/runs/rpg_controls/video23_l20_sentence_t5_opq4"

case "$mode" in
  mtp)
    objective=mtp_only
    backbone_lr=0.0003
    output="$root/rpg_mtp_unary"
    ;;
  unary)
    objective=catalog_plus_token
    backbone_lr=0.0001
    output="$root/catalog_ce_unary"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

mkdir -p "$output"
cd "$repo"

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
  --variant unary \
  --training-objective "$objective" \
  --mtp-temperature 0.07 \
  --token-loss-weight 0.1 \
  --epochs 30 \
  --patience 5 \
  --min-epochs 10 \
  --batch-size 256 \
  --eval-batch-size 128 \
  --backbone-lr "$backbone_lr" \
  --weight-decay 0.0001 \
  --proposal-k 72 \
  --selection-metric ndcg10 \
  --skip-ar-verifier \
  --seed 2026 \
  --output-dir "$output"
