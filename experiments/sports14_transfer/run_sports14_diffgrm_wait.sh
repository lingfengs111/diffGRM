#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-3}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/sports14_transfer"
processed="$repo/cache/AmazonReviews2014CleanGR/Sports_and_Outdoors/processed"
sem_ids="$processed/sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto.sem_ids"
mapping="$processed/item_id2tokens_sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto_4d.npy"
run_id=sports14_full_opq_cf_diff_guided_l20_long_v1
mkdir -p "$root"

# Avoid two quantizers racing on the same cache. The AR producer writes both
# artifacts before this independent guided-DiffGRM baseline starts.
while [[ ! -s "$sem_ids" || ! -s "$mapping" ]]; do
  sleep 20
done

cd "$repo"
CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=DIFF_GRM --dataset=AmazonReviews2014CleanGR \
  --config=experiments/amazon14_domains/common_l20.yaml \
  --config=experiments/amazon14_domains/sports14.yaml \
  --run_id="$run_id" >"$root/$run_id.log" 2>&1

