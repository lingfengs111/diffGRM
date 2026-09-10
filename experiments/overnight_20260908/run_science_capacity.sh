#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU width_d176|depth_d256|compact_d128}"
arm="${2:?usage: $0 GPU ARM}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python

case "$arm" in
  width_d176)
    config=experiments/overnight_20260908/science23_width_d176.yaml
    drafter_layers=4; pair_rank=35
    ;;
  depth_d256)
    config=experiments/overnight_20260908/science23_depth_d256.yaml
    drafter_layers=2; pair_rank=51
    ;;
  compact_d128)
    config=experiments/overnight_20260908/science23_compact_d128.yaml
    drafter_layers=2; pair_rank=32
    ;;
  *) echo "unknown capacity arm: $arm" >&2; exit 2 ;;
esac

root="$repo/runs/overnight_20260908/capacity/$arm"
ar_id="science23_opq4_${arm}_ar_l20_v1"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_id}/pytorch_model.bin"
result="$root/onepass_pairwise_ar/result.json"
mkdir -p "$root"
cd "$repo"

if [[ ! -s "$ar_ckpt" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
    --config=experiments/amazon23_domains/common.yaml \
    --config="$config" \
    --config=experiments/canonical_full/ar_constrained.yaml \
    --run_id="$ar_id" >"$root/ar.log" 2>&1
fi
test -s "$ar_ckpt"

if [[ ! -s "$result" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config "$config" \
    --ar-config experiments/canonical_full/ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    --ar-checkpoint "$ar_ckpt" \
    --backbone-architecture encoder_four_head \
    --backbone-initialization random --encoder-head-n-layer "$drafter_layers" \
    --variant pairwise --conditioner diffusion_encoder --pair-rank "$pair_rank" \
    --epochs 180 --patience 20 --min-epochs 16 \
    --batch-size 256 --eval-batch-size 32 \
    --backbone-lr 0.0003 --selector-lr 0.001 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k 72 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --retain-candidate-checkpoint --dump-selected-ranks \
    --output-dir "$root/onepass_pairwise_ar" >"$root/drafter.log" 2>&1
fi
test -s "$result"
printf 'complete %s %s\n' "$arm" "$(date --iso-8601=seconds)" >"$root/COMPLETE"
