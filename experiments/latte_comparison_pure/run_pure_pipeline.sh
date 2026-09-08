#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU science23_rqkmeans3|science23_opq3|science23_opq4|video23_opq4}"
arm="${2:?usage: $0 GPU ARM}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python

case "$arm" in
  science23_rqkmeans3)
    domain=science23; quantizer=rqkmeans; digits=3; splits=splits_l20
    data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_science/raw_core5
    processed="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed"
    sid="$processed/sentence-t5-base_latte-meta_pca192_RQKMEANS3x256_psid.sem_ids"
    ;;
  science23_opq3)
    domain=science23; quantizer=opq; digits=3; splits=splits_l20
    data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_science/raw_core5
    processed="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed"
    sid="$processed/sentence-t5-base_latte-meta_pca192_OPQ3x256_psid.sem_ids"
    ;;
  science23_opq4)
    domain=science23; quantizer=opq; digits=4; splits=splits_l20
    data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_science/raw_core5
    processed="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed"
    sid="$processed/sentence-t5-base_latte-meta_pca192_OPQ4x256_psid.sem_ids"
    ;;
  video23_opq4)
    domain=video23; quantizer=opq; digits=4; splits=splits
    data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_video_game/raw_core5
    processed="$repo/cache/AmazonReviews2023CleanGR/Video_Games/processed"
    sid="$processed/sentence-t5-base_latte-meta_pca192_OPQ4x256_psid.sem_ids"
    ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

root="$repo/runs/latte_comparison_pure/$arm"
embedding="$processed/sentence-t5-base_latte_meta_raw_d768.sent_emb"
sid_config="experiments/latte_comparison_pure/${arm}.yaml"
ar_run="${arm}_pure_ar_l20_v1"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_run}/pytorch_model.bin"
mkdir -p "$root"
cd "$repo"
test -s "$embedding"

if [[ ! -s "$sid" ]]; then
  test "$quantizer" = opq
  "$python_bin" scripts/generate_latte_rqkmeans_sids.py \
    --data-dir "$data" --splits-dir "$splits" \
    --embedding-path "$embedding" --output "$sid" \
    --embedding-dim 768 --pca-dim 192 --quantizer opq \
    --n-codebooks "$digits" --codebook-size 256 --faiss-threads 32 \
    >"$root/sid.log" 2>&1
fi
test -s "$sid"

if [[ ! -s "$ar_ckpt" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
    --config=experiments/amazon23_domains/common.yaml \
    --config="$sid_config" \
    --config=experiments/canonical_full/ar_constrained.yaml \
    --run_id="$ar_run" >"$root/ar.log" 2>&1
fi
test -s "$ar_ckpt"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --sid-config "$sid_config" \
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
