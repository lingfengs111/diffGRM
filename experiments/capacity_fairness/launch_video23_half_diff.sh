#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_root="$repo/runs/capacity_fairness/video23_half_2x2_d176"
run_id=video23_full_opq_cf_diff_guided_half_2x2_d176_v1
mkdir -p "$run_root"
cd "$repo"

exec env TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=DIFF_GRM \
  --dataset=AmazonReviews2023CleanGR \
  --config=experiments/canonical_full/video23_cf_official.yaml \
  --config=experiments/capacity_fairness/video23_half_2x2_d176.yaml \
  --run_id="$run_id" \
  >"$run_root/$run_id.log" 2>&1

