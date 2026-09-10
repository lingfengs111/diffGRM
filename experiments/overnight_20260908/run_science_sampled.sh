#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU N_NEGATIVES}"
n_negatives="${2:?usage: $0 GPU N_NEGATIVES}"
case "$n_negatives" in 256|1024|4096) ;; *) echo "unsupported K=$n_negatives" >&2; exit 2;; esac

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/overnight_20260908/sampled_catalog/k${n_negatives}"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_science23_opq4_pure_ar_l20_v1/pytorch_model.bin"
mkdir -p "$root"
cd "$repo"
test -s "$ar_ckpt"

if [[ ! -s "$root/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config experiments/latte_comparison_pure/science23_opq4.yaml \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --ar-checkpoint "$ar_ckpt" \
    --backbone-architecture encoder_four_head \
    --backbone-initialization random --encoder-head-n-layer 4 \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
    --training-objective catalog_plus_token \
    --sampled-catalog-negatives "$n_negatives" \
    --epochs 180 --patience 20 --min-epochs 16 \
    --batch-size 256 --eval-batch-size 32 \
    --backbone-lr 0.0003 --selector-lr 0.001 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k 72 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --retain-candidate-checkpoint --dump-selected-ranks \
    --output-dir "$root" >"$root/train.log" 2>&1
fi
test -s "$root/result.json"
printf 'complete K=%s %s\n' "$n_negatives" "$(date --iso-8601=seconds)" >"$root/COMPLETE"
