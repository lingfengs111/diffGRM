#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU video23|music23}"
domain="${2:?usage: $0 GPU video23|music23}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/overnight_20260908/dual_view/$domain"
mkdir -p "$root"
cd "$repo"

case "$domain" in
  video23)
    rq_sid=experiments/overnight_20260908/video23_rqkmeans3.yaml
    opq_sid=experiments/latte_comparison_pure/video23_opq4.yaml
    rq_ar_id=video23_rqkmeans3_pure_ar_l20_overnight_v1
    opq_ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_video23_opq4_pure_ar_l20_v1/pytorch_model.bin"
    opq_drafter_ckpt="$repo/runs/latte_comparison_pure/video23_opq4/onepass_pairwise_ar/best.pt"
    ;;
  music23)
    rq_sid=experiments/overnight_20260908/music23_rqkmeans3_sameview.yaml
    opq_sid=experiments/music23_transfer/music23_l20_long.yaml
    rq_ar_id=music23_rqkmeans3_sameview_ar_l20_overnight_v1
    opq_ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_music23_full_opq_cf_ar_l20_long_v1/pytorch_model.bin"
    opq_drafter_ckpt="$repo/runs/music23_transfer/random_encoder4_pairwise_ar/best.pt"
    rq_sid_path="$repo/cache/AmazonReviews2023CleanGR/Musical_Instruments/processed/sentence-t5-base_pca256_RQKMEANS3x256_sameview_esm.sem_ids"
    if [[ ! -s "$rq_sid_path" ]]; then
      "$python_bin" scripts/generate_latte_rqkmeans_sids.py \
        --data-dir /home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_music/raw_core5 \
        --splits-dir splits \
        --embedding-path "$repo/cache/AmazonReviews2023CleanGR/Musical_Instruments/processed/sentence-t5-base_pca256.sent_emb" \
        --embedding-dim 256 --pca-dim 256 --pretransformed-input \
        --quantizer rqkmeans --n-codebooks 3 --codebook-size 256 \
        --repair-strategy esm --esm-neighbors 5 --faiss-threads 16 \
        --raw-codes-output "$root/music23_rqk3_raw.npy" \
        --output "$rq_sid_path" >"$root/prepare_rqk3.log" 2>&1
    fi
    ;;
  *) echo "unknown domain: $domain" >&2; exit 2 ;;
esac

rq_ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${rq_ar_id}/pytorch_model.bin"
rq_drafter_dir="$root/rqk3_matched_drafter"
rq_drafter_ckpt="$rq_drafter_dir/best.pt"

test -s "$opq_ar_ckpt"
test -s "$opq_drafter_ckpt"

evaluate_dual() {
  local name="$1" drafter_sid="$2" verifier_sid="$3"
  local drafter_ckpt="$4" ar_ckpt="$5"
  local output="$root/$name/result.json"
  mkdir -p "$root/$name"
  if [[ -s "$output" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/evaluate_dual_view_fusion.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --drafter-sid-config "$drafter_sid" \
    --verifier-sid-config "$verifier_sid" \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --drafter-checkpoint "$drafter_ckpt" --ar-checkpoint "$ar_ckpt" \
    --proposal-k 72 --eval-batch-size 32 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --output "$output" >"$root/$name/eval.log" 2>&1
}

if [[ ! -s "$rq_ar_ckpt" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
    --config=experiments/amazon23_domains/common.yaml \
    --config="$rq_sid" \
    --config=experiments/canonical_full/ar_constrained.yaml \
    --run_id="$rq_ar_id" >"$root/rqk3_ar.log" 2>&1
fi
test -s "$rq_ar_ckpt"

# This is the main hypothesis and does not need to wait for the reverse-view
# drafter to finish training.
evaluate_dual opq4_to_rqk3 "$opq_sid" "$rq_sid" "$opq_drafter_ckpt" "$rq_ar_ckpt"

if [[ ! -s "$rq_drafter_dir/result.json" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config "$rq_sid" \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --ar-checkpoint "$rq_ar_ckpt" \
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
    --output-dir "$rq_drafter_dir" >"$root/rqk3_drafter.log" 2>&1
fi
test -s "$rq_drafter_ckpt"
evaluate_dual rqk3_to_opq4 "$rq_sid" "$opq_sid" "$rq_drafter_ckpt" "$opq_ar_ckpt"
printf 'complete %s %s\n' "$domain" "$(date --iso-8601=seconds)" >"$root/COMPLETE"
