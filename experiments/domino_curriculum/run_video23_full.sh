#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
mode="${1:?usage: run_video23_full.sh baseline|strict|persistent}"
root="$repo/runs/domino_curriculum/video23_l20_full_20260902"

case "$mode" in
  baseline)
    anchor_start=0.0
    anchor_end=0.0
    ;;
  strict)
    # Domino's paper schedule: base-only at the beginning, final-only at end.
    anchor_start=1.0
    anchor_end=0.0
    ;;
  persistent)
    # Keep a small final unary anchor to test whether permanent protection is
    # preferable when candidate recall, rather than accepted-prefix length, is
    # the downstream objective.
    anchor_start=1.0
    anchor_end=0.2
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

output="$root/$mode"
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
  --backbone-architecture masked_decoder \
  --backbone-initialization diffusion_pretrained \
  --conditioner diffusion_encoder \
  --variant pairwise \
  --pair-rank 51 \
  --training-objective catalog_plus_token \
  --token-loss-weight 0.1 \
  --base-anchor-start "$anchor_start" \
  --base-anchor-end "$anchor_end" \
  --base-anchor-decay-epochs 20 \
  --epochs 20 \
  --patience 5 \
  --min-epochs 20 \
  --batch-size 256 \
  --eval-batch-size 64 \
  --backbone-lr 0.0003 \
  --selector-lr 0.001 \
  --weight-decay 0.0001 \
  --proposal-k 72 \
  --selection-metric candidate_recall \
  --skip-ar-verifier \
  --seed 2026 \
  --output-dir "$output"
