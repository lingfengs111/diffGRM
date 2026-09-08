#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
output="$repo/runs/domino_causal_corrector/video23_full_20260902"

mkdir -p "$output"
cd "$repo"

exec env TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  "$python_bin" scripts/train_domino_causal_corrector.py \
  --base-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --epochs 3 \
  --patience 1 \
  --min-epochs 1 \
  --batch-size 256 \
  --eval-batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --residual-l2 0.00001 \
  --state-dim 64 \
  --correction-rank 32 \
  --pool-k 128 \
  --proposal-k 72 \
  --candidate-chunk-size 32 \
  --ar-chunk-size 16 \
  --correction-betas 0,0.1,0.25,0.5,0.75,1 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --selection-metric final_ndcg10 \
  --seed 2026 \
  --output-dir "$output"
