#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
output="$repo/runs/rank32_handoff/video23_medium_20260903"

mkdir -p "$output"
cd "$repo"

exec env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  "$python_bin" scripts/train_domino_causal_corrector.py \
  --base-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --epochs 2 \
  --patience 1 \
  --min-epochs 1 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --residual-l2 0.00001 \
  --token-nll-weight 0 \
  --candidate-listwise-weight 0 \
  --ar-residual-distill-weight 0 \
  --rank-aware-topk-weight 1.0 \
  --rank-aware-min-rank 11 \
  --rank-aware-max-rank 32 \
  --rank-aware-top-k 10 \
  --head-preservation-weight 0.25 \
  --head-preservation-k 10 \
  --train-candidates 32 \
  --boundary-hard-fraction 0.6 \
  --hard-boundary-rank 32 \
  --teacher-fusion-alpha 1.0 \
  --state-dim 64 \
  --correction-rank 32 \
  --pool-k 128 \
  --proposal-k 72 \
  --candidate-chunk-size 32 \
  --ar-chunk-size 8 \
  --correction-betas 0,0.025,0.05,0.1,0.15,0.25,0.5,0.75,1 \
  --fusion-alphas 0,0.1,0.25,0.5,0.65,0.75,0.85,0.9,1 \
  --selection-metric final_ndcg10 \
  --recall72-tolerance 0 \
  --max-train-examples 65536 \
  --max-val-examples 16384 \
  --max-test-examples 16384 \
  --seed 2026 \
  --output-dir "$output"
