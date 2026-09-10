#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: run_arm.sh GPU ARM smoke|full}"
arm="${2:?usage: run_arm.sh GPU ARM smoke|full}"
stage="${3:?usage: run_arm.sh GPU ARM smoke|full}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/verifier_objectives_20260909/$stage/$arm"

common=(
  --dataset AmazonReviews2023CleanGR
  --common-config experiments/amazon23_domains/common.yaml
  --sid-config experiments/latte_comparison_pure/video23_opq4.yaml
  --ar-config experiments/canonical_full/ar_constrained.yaml
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml
  --drafter-checkpoint runs/latte_comparison_pure/video23_opq4/onepass_pairwise_ar/best.pt
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_opq4_pure_ar_l20_v1/pytorch_model.bin
  --batch-size 64 --eval-batch-size 32
  --learning-rate 0.00001 --weight-decay 0.0001
  --trainable-scope last_decoder
  --token-loss-weight 1.0
  --listwise-temperature 1.0 --margin-weight 0.0 --ranking-score ar
  --num-negatives 15 --candidate-score-chunk-size 8
  --proposal-k 72 --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1
  --seed 2026 --output-dir "$root"
)

case "$arm" in
  apao_all)
    objective=(
      --listwise-weight 0.0 --teacher-distill-weight 1.0
      --prefix-rank-weight 0.1 --prefix-rank-temperature 1.0
      --prefix-adaptive-eta 0.0001 --prefix-negative-scale
      --negative-strata 8,4,3
    )
    ;;
  apao_support)
    objective=(
      --listwise-weight 0.0 --teacher-distill-weight 1.0
      --prefix-rank-weight 0.1 --prefix-rank-temperature 1.0
      --prefix-adaptive-eta 0.0001 --prefix-negative-scale
      --negative-strata 8,4,3 --require-target-in-proposals
    )
    ;;
  ar_hard_support)
    objective=(
      --listwise-weight 0.05 --teacher-distill-weight 1.0
      --negative-mining current_ar --require-target-in-proposals
    )
    ;;
  lambda_dpo_support)
    objective=(
      --listwise-weight 0.0 --teacher-distill-weight 0.0
      --preference-weight 0.1 --preference-beta 0.5
      --preference-ndcg-k 10 --training-fusion-alpha 0.75
      --negative-strata 8,4,3 --require-target-in-proposals
    )
    ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

if [[ "$stage" == smoke ]]; then
  schedule=(
    --epochs 1 --patience 1 --min-epochs 1
    --max-train-examples 512 --max-val-examples 256 --max-test-examples 256
  )
elif [[ "$stage" == full ]]; then
  schedule=(--epochs 6 --patience 2 --min-epochs 2)
else
  echo "unknown stage: $stage" >&2
  exit 2
fi

mkdir -p "$root"
cd "$repo"
if [[ -s "$root/result.json" ]]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$python_bin" scripts/train_candidate_aware_verifier.py \
  "${common[@]}" "${objective[@]}" "${schedule[@]}" \
  >"$root/train.log" 2>&1

test -s "$root/result.json"
date --iso-8601=seconds >"$root/COMPLETE"
