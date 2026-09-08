#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_id=music23_full_opq_cf_ar_l20_long_v1
root="$repo/runs/music23_transfer"
mkdir -p "$root"
cd "$repo"

env TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
  --config=experiments/amazon23_domains/common.yaml \
  --config=experiments/music23_transfer/music23_l20_long.yaml \
  --config=experiments/canonical_full/video23_ar_constrained.yaml \
  --run_id="$run_id" >"$root/$run_id.log" 2>&1
