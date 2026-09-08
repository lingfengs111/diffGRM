#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_id=science23_full_opq_cf_diff_guided_l20_long_v1
root="$repo/runs/science23_transfer"
processed="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed"
sem_ids="$processed/sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto.sem_ids"
mapping="$processed/item_id2tokens_sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto_4d.npy"
mkdir -p "$root"
test -s "$sem_ids"
test -s "$mapping"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=DIFF_GRM --dataset=AmazonReviews2023CleanGR \
  --config=experiments/amazon23_domains/common.yaml \
  --config=experiments/amazon23_domains/science23_l20_long.yaml \
  --run_id="$run_id" >"$root/$run_id.log" 2>&1

