#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_lambda_repeat.sh GPU SEED}"
seed="${2:?usage: run_lambda_repeat.sh GPU SEED}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/verifier_objectives_20260909/lambda_dpo_repeats_e12/seed_$seed"

mkdir -p "$root"
cd "$repo"
if [[ -s "$root/result.json" ]]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$python_bin" scripts/train_candidate_aware_verifier.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --sid-config experiments/latte_comparison_pure/video23_opq4.yaml \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --drafter-checkpoint runs/latte_comparison_pure/video23_opq4/onepass_pairwise_ar/best.pt \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_opq4_pure_ar_l20_v1/pytorch_model.bin \
  --batch-size 64 --eval-batch-size 32 \
  --learning-rate 0.00001 --weight-decay 0.0001 \
  --trainable-scope last_decoder \
  --token-loss-weight 1.0 \
  --listwise-weight 0.0 --margin-weight 0.0 \
  --teacher-distill-weight 0.0 \
  --preference-weight 0.1 --preference-beta 0.5 \
  --preference-ndcg-k 10 --training-fusion-alpha 0.75 \
  --ranking-score ar --listwise-temperature 1.0 \
  --num-negatives 15 --negative-strata 8,4,3 \
  --require-target-in-proposals --candidate-score-chunk-size 8 \
  --proposal-k 72 --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --epochs 12 --patience 4 --min-epochs 8 \
  --seed "$seed" --output-dir "$root" \
  >"$root/train.log" 2>&1

test -s "$root/result.json"
date --iso-8601=seconds >"$root/COMPLETE"
