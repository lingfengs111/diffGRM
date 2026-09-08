#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU science23|video23}"
domain="${2:?usage: $0 GPU science23|video23}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
model=/home/lingfengs111/.cache/huggingface/hub/models--sentence-transformers--sentence-t5-base/snapshots/fc5d4628481afbbaaacd7af6bb07cf9d3865f781

case "$domain" in
  science23)
    data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_science/raw_core5
    splits=splits_l20
    category=Industrial_and_Scientific
    processed="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed"
    epochs=180
    patience=18
    ;;
  video23)
    data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_video_game/raw_core5
    splits=splits
    category=Video_Games
    processed="$repo/cache/AmazonReviews2023CleanGR/Video_Games/processed"
    epochs=160
    patience=16
    ;;
  *) echo "unknown domain: $domain" >&2; exit 2 ;;
esac

root="$repo/runs/latte_aligned/$domain/opq3_tokenizer_control"
embedding="$processed/sentence-t5-base_latte_meta_raw_d768.sent_emb"
sid="$processed/sentence-t5-base_latte-meta_pca192_OPQ3x256_psid.sem_ids"
sid_config="experiments/latte_aligned/${domain}_opq3_latent8.yaml"
ar_run="${domain}_latte_opq3_latent8_ar_l20_v1"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_run}/pytorch_model.bin"
mkdir -p "$root"
cd "$repo"
test -s "$embedding"

if [[ ! -s "$sid" ]]; then
  "$python_bin" scripts/generate_latte_rqkmeans_sids.py \
    --data-dir "$data" --splits-dir "$splits" \
    --embedding-path "$embedding" --output "$sid" \
    --embedding-dim 768 --pca-dim 192 --quantizer opq \
    --n-codebooks 3 --codebook-size 256 --faiss-threads 32 \
    >"$root/sid.log" 2>&1
fi
test -s "$sid"
test -s "$sid.diagnostics.json"

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
