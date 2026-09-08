#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/toys14_transfer_l50/exact_mips_id_ar"
ar_ckpt="$repo/saved/AmazonReviews2014CleanGR_toys14_full_opq_cf_ar_l50_long_v1/pytorch_model.bin"
mkdir -p "$root"
test -s "$ar_ckpt"
cd "$repo"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_ann_drafter.py \
  --dataset AmazonReviews2014CleanGR \
  --common-config experiments/amazon14_domains/common_l50.yaml \
  --ar-config experiments/toys14_transfer/toys14_l50_ar_mips.yaml \
  --ar-checkpoint "$ar_ckpt" --item-mode id \
  --epochs 10 --batch-size 256 --eval-batch-size 64 \
  --lr 0.001 --weight-decay 0.0001 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$root" >"$root/train.log" 2>&1
