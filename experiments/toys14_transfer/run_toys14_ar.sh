#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_id=toys14_full_opq_cf_ar_l20_long_v1
root="$repo/runs/toys14_transfer"
mkdir -p "$root"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=AR_GRM --dataset=AmazonReviews2014CleanGR \
  --config=experiments/amazon14_domains/common_l20.yaml \
  --config=experiments/amazon14_domains/toys14.yaml \
  --config=experiments/canonical_full/video23_ar_constrained.yaml \
  --run_id="$run_id" >"$root/$run_id.log" 2>&1
