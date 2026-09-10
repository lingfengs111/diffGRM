#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU last_decoder|decoder}"
arm="${2:?usage: $0 GPU ARM}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python

case "$arm" in
  last_decoder)
    scope=last_decoder; lr=0.000003; listwise=0.05; distill=1.0
    ;;
  decoder)
    scope=decoder; lr=0.000001; listwise=0.10; distill=2.0
    ;;
  *) echo "unknown proposal-aware arm: $arm" >&2; exit 2 ;;
esac

root="$repo/runs/verifier_next_20260909/proposal_aware_video23/$arm"
mkdir -p "$root"
cd "$repo"

if [[ ! -s "$root/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_candidate_aware_verifier.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config experiments/latte_comparison_pure/video23_opq4.yaml \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --drafter-checkpoint runs/latte_comparison_pure/video23_opq4/onepass_pairwise_ar/best.pt \
    --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_opq4_pure_ar_l20_v1/pytorch_model.bin \
    --epochs 8 --patience 3 --min-epochs 3 \
    --batch-size 64 --eval-batch-size 32 \
    --learning-rate "$lr" --weight-decay 0.0001 \
    --trainable-scope "$scope" \
    --token-loss-weight 1.0 \
    --listwise-weight "$listwise" --listwise-temperature 1.0 \
    --teacher-distill-weight "$distill" --teacher-temperature 1.0 \
    --margin-weight 0.0 --ranking-score ar \
    --require-target-in-proposals --negative-strata 8,4,3 \
    --head-rank-weight 0.25 --middle-rank-weight 1.0 --tail-rank-weight 1.5 \
    --num-negatives 15 --candidate-score-chunk-size 8 \
    --proposal-k 72 --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --seed 2026 --output-dir "$root" >"$root/train.log" 2>&1
fi
test -s "$root/result.json"
date --iso-8601=seconds >"$root/COMPLETE"
