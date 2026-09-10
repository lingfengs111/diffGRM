#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU deep_d256|wide_d320}"
arm="${2:?usage: $0 GPU ARM}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
sid_config=experiments/overnight_20260908/science23_width_d176.yaml
drafter_ckpt=runs/overnight_20260908/capacity/width_d176/onepass_pairwise_ar/best.pt

case "$arm" in
  deep_d256)
    ar_config=experiments/verifier_next_20260909/science_ar_deep_d256.yaml
    run_id=science23_opq4_small_drafter_deep_d256_ar_l20_v1
    ;;
  wide_d320)
    ar_config=experiments/verifier_next_20260909/science_ar_wide_d320.yaml
    run_id=science23_opq4_small_drafter_wide_d320_ar_l20_v1
    ;;
  *) echo "unknown capacity arm: $arm" >&2; exit 2 ;;
esac

root="$repo/runs/verifier_next_20260909/capacity_reallocation/$arm"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${run_id}/pytorch_model.bin"
mkdir -p "$root"
cd "$repo"

if [[ ! -s "$ar_ckpt" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
    --config=experiments/amazon23_domains/common.yaml \
    --config="$sid_config" --config="$ar_config" \
    --run_id="$run_id" >"$root/ar_train.log" 2>&1
fi
test -s "$ar_ckpt"

if [[ ! -s "$root/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/evaluate_dual_view_fusion.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --drafter-sid-config "$sid_config" \
    --verifier-sid-config "$sid_config" \
    --ar-config "$ar_config" \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --drafter-checkpoint "$drafter_ckpt" \
    --ar-checkpoint "$ar_ckpt" \
    --proposal-k 72 --eval-batch-size 32 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --seed 2026 --output "$root/result.json" \
    >"$root/eval.log" 2>&1
fi
test -s "$root/result.json"
date --iso-8601=seconds >"$root/COMPLETE"
