#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
out="$repo/runs/verifier_co_design/video23_guarded_ar_residual_k72"
mkdir -p "$out"
cd "$repo"

# The frozen AR score is the exact epoch-0 baseline.  A zero-initialized
# lightweight bidirectional scorer may only add a candidate-aware residual.
# Training uses the drafter's real top-72 set; it does not leak the target into
# queries where proposal retrieval actually missed it.
env TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "$python_bin" \
  scripts/train_parallel_path_verifier.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --drafter-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --teacher-ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin \
  --pair-rank 51 --proposal-k 72 --num-negatives 71 \
  --residual-over-teacher-ar --train-on-proposal-set \
  --coordinate-mode bidirectional --history-pooling last \
  --hidden-dim 128 --n-head 4 --coordinate-layers 1 --set-layers 1 \
  --label-weight 1.0 --distill-weight 0.5 --distill-temperature 1.0 \
  --margin-weight 0.1 --margin-value 0.2 \
  --candidate-score-chunk-size 24 \
  --epochs 6 --patience 2 --batch-size 64 --eval-batch-size 64 \
  --learning-rate 0.0001 --weight-decay 0.0001 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$out" \
  >"$out/train.log" 2>&1
