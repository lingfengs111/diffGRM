#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
reference="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed/sentence-t5-base_latte-meta_pca192_OPQ4x256_psid.sem_ids"
random_sid="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed/science23_OPQ4x256_psid_item_permutation_seed2026.sem_ids"
config=experiments/sid_semantics_20260909/science23_random_permuted_opq4.yaml
root="$repo/runs/sid_semantics_20260909/science23_random_permuted_opq4"
ar_run=science23_random_permuted_opq4_ar_l20_seed2026
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_run}/pytorch_model.bin"
mkdir -p "$root"
cd "$repo"

if [[ ! -s "$random_sid" ]]; then
  "$python_bin" scripts/permute_sid_assignments.py \
    --reference "$reference" --output "$random_sid" --seed 2026 \
    >"$root/sid_permutation.log" 2>&1
fi
test -s "$random_sid"

if [[ ! -s "$ar_ckpt" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
    --config=experiments/amazon23_domains/common.yaml \
    --config="$config" \
    --config=experiments/canonical_full/ar_constrained.yaml \
    --run_id="$ar_run" >"$root/ar.log" 2>&1
fi
test -s "$ar_ckpt"

if [[ ! -s "$root/onepass_pairwise_ar/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config "$config" \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --ar-checkpoint "$ar_ckpt" \
    --backbone-architecture encoder_four_head \
    --backbone-initialization random --encoder-head-n-layer 4 \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
    --epochs 180 --patience 20 --min-epochs 16 \
    --batch-size 256 --eval-batch-size 32 \
    --backbone-lr 0.0003 --selector-lr 0.001 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k 72 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --retain-candidate-checkpoint --dump-selected-ranks \
    --output-dir "$root/onepass_pairwise_ar" >"$root/onepass.log" 2>&1
fi
test -s "$root/onepass_pairwise_ar/result.json"
date --iso-8601=seconds >"$root/COMPLETE"
