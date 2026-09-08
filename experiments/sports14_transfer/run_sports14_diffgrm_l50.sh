#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-2}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/sports14_transfer_l50"
run_id=sports14_full_opq_cf_diff_guided_l50_long_v1
mkdir -p "$root"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=DIFF_GRM --dataset=AmazonReviews2014CleanGR \
  --config=experiments/amazon14_domains/common_l50.yaml \
  --config=experiments/amazon14_domains/sports14_l50.yaml \
  --run_id="$run_id" >"$root/$run_id.log" 2>&1

