#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU opq4_to_rqk3|rqk3_to_opq4}"
arm="${2:?usage: $0 GPU ARM}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python

case "$arm" in
  opq4_to_rqk3)
    drafter_sid=experiments/latte_comparison_pure/science23_opq4.yaml
    drafter_ckpt="$repo/runs/latte_comparison_pure/science23_opq4/onepass_pairwise_ar/best.pt"
    verifier_sid=experiments/latte_comparison_pure/science23_rqkmeans3.yaml
    ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_science23_rqkmeans3_pure_ar_l20_v1/pytorch_model.bin"
    ;;
  rqk3_to_opq4)
    drafter_sid=experiments/latte_comparison_pure/science23_rqkmeans3.yaml
    drafter_ckpt="$repo/runs/latte_comparison_pure/science23_rqkmeans3/onepass_pairwise_ar/best.pt"
    verifier_sid=experiments/latte_comparison_pure/science23_opq4.yaml
    ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_science23_opq4_pure_ar_l20_v1/pytorch_model.bin"
    ;;
  *) echo "unknown dual-view arm: $arm" >&2; exit 2 ;;
esac

root="$repo/runs/tokenizer_controls_20260907/dual_view/$arm"
output="$root/result_v2.json"
mkdir -p "$root"
cd "$repo"
test -s "$drafter_ckpt"
test -s "$ar_ckpt"
CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/evaluate_dual_view_fusion.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --drafter-sid-config "$drafter_sid" \
  --verifier-sid-config "$verifier_sid" \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --drafter-checkpoint "$drafter_ckpt" \
  --ar-checkpoint "$ar_ckpt" \
  --proposal-k 72 --eval-batch-size 32 --seed 2026 \
  --output "$output" >"$root/eval.log" 2>&1
test -s "$output"
