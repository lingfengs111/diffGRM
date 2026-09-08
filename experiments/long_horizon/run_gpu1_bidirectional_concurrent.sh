#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/long_horizon_20260831"
mkdir -p "$root"
cd "$repo"

exec env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_path_verifier.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --drafter-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --init-verifier-checkpoint runs/verifier_only/video23_full_20260830/bidirectional_last_k72_random/best.pt \
  --pair-rank 51 --proposal-k 72 \
  --epochs 20 --patience 4 --batch-size 64 --eval-batch-size 64 \
  --num-negatives 71 --hidden-dim 128 --n-head 4 \
  --coordinate-layers 1 --set-layers 1 \
  --coordinate-mode bidirectional --history-pooling last \
  --learning-rate 0.0001 --weight-decay 0.0001 \
  --label-weight 1.0 --distill-weight 0.0 \
  --margin-weight 0.1 --margin-value 0.2 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$root/bidirectional_verifier_converged" \
  >"$root/bidirectional_verifier_converged.log" 2>&1

