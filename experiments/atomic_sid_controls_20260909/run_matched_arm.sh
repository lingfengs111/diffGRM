#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_matched_arm.sh GPU atomic|unary|pairwise}"
arm="${2:?usage: run_matched_arm.sh GPU atomic|unary|pairwise}"
case "$arm" in
  atomic|unary|pairwise) ;;
  *) echo "invalid arm: $arm" >&2; exit 2 ;;
esac

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
output="$repo/runs/atomic_sid_controls_20260909/science23_matched_${arm}"
mkdir -p "$output"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/train_matched_catalog_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --sid-config experiments/latte_comparison_pure/science23_opq4.yaml \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_science23_opq4_pure_ar_l20_v1/pytorch_model.bin \
  --representation "$arm" --pair-rank 32 \
  --epochs 16 --batch-size 256 --eval-batch-size 32 \
  --lr 0.001 --weight-decay 0.0001 \
  --proposal-ks 72,128 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$output" \
  2>&1 | tee "$output/train.log"
