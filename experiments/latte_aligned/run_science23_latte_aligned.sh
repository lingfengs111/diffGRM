#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU [ar|onepass|all]}"
mode="${2:-all}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/latte_aligned/science23"
sid="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed/sentence-t5-base_latte-meta_pca192_RQKMEANS3x256_psid.sem_ids"
ar_run=science23_latte_rqkmeans3_latent8_ar_l20_v1
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_run}/pytorch_model.bin"
mkdir -p "$root"
test -s "$sid"
cd "$repo"

run_ar() {
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
    --config=experiments/amazon23_domains/common.yaml \
    --config=experiments/latte_aligned/science23_rqkmeans3_latent8.yaml \
    --config=experiments/canonical_full/ar_constrained.yaml \
    --run_id="$ar_run" >"$root/ar.log" 2>&1
}

run_onepass() {
  test -s "$ar_ckpt"
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config experiments/latte_aligned/science23_rqkmeans3_latent8.yaml \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --ar-checkpoint "$ar_ckpt" \
    --backbone-architecture encoder_four_head \
    --backbone-initialization random --encoder-head-n-layer 4 \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
    --epochs 80 --patience 14 --min-epochs 16 \
    --batch-size 256 --eval-batch-size 32 \
    --backbone-lr 0.0003 --selector-lr 0.001 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k 72 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --output-dir "$root/onepass_pairwise_ar" >"$root/onepass.log" 2>&1
}

case "$mode" in
  ar) run_ar ;;
  onepass) run_onepass ;;
  all) run_ar; run_onepass ;;
  *) echo "usage: $0 GPU [ar|onepass|all]" >&2; exit 2 ;;
esac

