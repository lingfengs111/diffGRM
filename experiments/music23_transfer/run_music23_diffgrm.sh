#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_id=music23_full_opq_cf_diff_guided_l20_long_v1
root="$repo/runs/music23_transfer"
processed="$repo/cache/AmazonReviews2023CleanGR/Musical_Instruments/processed"
sem_ids="$processed/sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto.sem_ids"
mapping="$processed/item_id2tokens_sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto_4d.npy"
mkdir -p "$root"
cd "$repo"

# The dynamic scheduler launches this job only after the AR cache producer has
# atomically completed both files. Refuse an unsafe direct launch instead of
# racing two writers in the shared processed directory.
test -s "$sem_ids"
test -s "$mapping"

env TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=DIFF_GRM --dataset=AmazonReviews2023CleanGR \
  --config=experiments/amazon23_domains/common.yaml \
  --config=experiments/music23_transfer/music23_l20_long.yaml \
  --run_id="$run_id" >"$root/$run_id.log" 2>&1
