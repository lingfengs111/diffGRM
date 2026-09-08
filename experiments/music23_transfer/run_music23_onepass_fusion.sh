#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/music23_transfer"
mode="${1:-all}"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_music23_full_opq_cf_ar_l20_long_v1/pytorch_model.bin"
diff_ckpt="$repo/saved/AmazonReviews2023CleanGR_music23_full_opq_cf_diff_guided_l20_long_v1/pytorch_model.bin"
mkdir -p "$root"
cd "$repo"
test -s "$ar_ckpt"

run_onepass() {
  local name="$1"
  local architecture="$2"
  local initialization="$3"
  shift 3
  local checkpoint_args=()
  if [[ "$initialization" == "diffusion_pretrained" ]]; then
    test -s "$diff_ckpt"
    checkpoint_args=(--diffusion-checkpoint "$diff_ckpt")
  fi
  env TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/amazon23_domains/common.yaml \
    --sid-config experiments/music23_transfer/music23_l20_long.yaml \
    --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
    --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
    "${checkpoint_args[@]}" --ar-checkpoint "$ar_ckpt" \
    --backbone-architecture "$architecture" \
    --backbone-initialization "$initialization" \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
    --epochs 60 --patience 10 --min-epochs 15 \
    --batch-size 256 --eval-batch-size 64 \
    --backbone-lr 0.0003 --selector-lr 0.001 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k 72 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    "$@" --output-dir "$root/$name" >"$root/$name.log" 2>&1
}

case "$mode" in
  random)
    # Primary proposed system: no DiffGRM initialization is required.
    run_onepass random_encoder4_pairwise_ar \
      encoder_four_head random --encoder-head-n-layer 4
    ;;
  pretrained)
    # Transfer control for the Video23 finding that denoising pretraining
    # helps but is not necessary.
    run_onepass diff_pretrained_masked_pairwise_ar \
      masked_decoder diffusion_pretrained
    ;;
  all)
    run_onepass random_encoder4_pairwise_ar \
      encoder_four_head random --encoder-head-n-layer 4
    run_onepass diff_pretrained_masked_pairwise_ar \
      masked_decoder diffusion_pretrained
    ;;
  *)
    echo "usage: $0 {random|pretrained|all}" >&2
    exit 2
    ;;
esac
