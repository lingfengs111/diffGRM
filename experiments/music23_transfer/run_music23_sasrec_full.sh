#!/usr/bin/env bash
set -euo pipefail

clean_repo=/home/lingfengs111/codes/GR/CleanGR
python_bin=/home/lingfengs111/.conda/envs/py313/bin/python
config=configs/amazon23_music_sasrec_long.yaml
root=/home/lingfengs111/codes/GR_variant/DiffGRM/runs/music23_transfer
mkdir -p "$root"
cd "$clean_repo"

env TOKENIZERS_PARALLELISM=false "$python_bin" -m cleangr.train.train_sasrec \
  --config "$config" >"$root/music23_sasrec_full_train.log" 2>&1
env TOKENIZERS_PARALLELISM=false "$python_bin" -m cleangr.evaluation.eval_sasrec \
  --config "$config" --split test --ks 5,10,20,50 --mask-history \
  >"$root/music23_sasrec_full_test.log" 2>&1
