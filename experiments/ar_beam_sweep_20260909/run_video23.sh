#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
checkpoint="$repo/saved/AmazonReviews2023CleanGR_video23_opq4_pure_ar_l20_v1/pytorch_model.bin"
root="$repo/runs/ar_beam_sweep_20260909/video23_opq4_standalone"
mkdir -p "$root"
cd "$repo"

for width in 256 500; do
  arm="$root/beam${width}"
  mkdir -p "$arm"
  if [[ ! -s "$arm/COMPLETE" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
      --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
      --checkpoint="$checkpoint" \
      --config=experiments/amazon23_domains/common.yaml \
      --config=experiments/latte_comparison_pure/video23_opq4.yaml \
      --config=experiments/canonical_full/ar_constrained.yaml \
      --config="experiments/ar_beam_sweep_20260909/beam${width}.yaml" \
      --run_id="video23_opq4_ar_beam${width}_eval_only" \
      >"$arm/eval.log" 2>&1
    rg 'Test Results:' "$arm/eval.log" | tail -1 >"$arm/test_result.txt"
    test -s "$arm/test_result.txt"
    date --iso-8601=seconds >"$arm/COMPLETE"
  fi
done
date --iso-8601=seconds >"$root/COMPLETE"
