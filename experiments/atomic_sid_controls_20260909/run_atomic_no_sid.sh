#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_atomic_no_sid.sh GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
output="$repo/runs/atomic_sid_controls_20260909/science23_atomic_no_sid"
mkdir -p "$output"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" \
  /home/lingfengs111/miniconda3/envs/diffgrm/bin/python \
  scripts/train_atomic_candidate_ranker.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --domain-config experiments/latte_comparison_pure/science23_opq4.yaml \
  --hidden-dim 256 --n-layer 2 --n-head 4 --n-inner 512 --dropout 0.1 \
  --retriever-epochs 20 --ranker-epochs 10 \
  --batch-size 256 --eval-batch-size 128 \
  --retriever-lr 0.0003 --ranker-lr 0.001 --weight-decay 0.0001 \
  --proposal-ks 72,128 --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$output" \
  2>&1 | tee "$output/train.log"
