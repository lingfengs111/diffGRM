#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/long_horizon_20260831"
common=experiments/canonical_full/video23_cf_official.yaml
ar_config=experiments/canonical_full/video23_ar_constrained.yaml
diff_config=experiments/canonical_full/diffusion_sequential.yaml
diff_ckpt=saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin
ar_ckpt=saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin

# This queue is intentionally device-agnostic.  The caller assigns a free GPU
# with CUDA_VISIBLE_DEVICES; neither control depends on the half-size jobs.
mkdir -p "$root"
cd "$repo"

run_control() {
  local name="$1"
  local init_checkpoint="$2"
  local architecture="$3"
  local initialization="$4"
  local variant="$5"
  local conditioner="$6"
  local pair_rank="$7"
  shift 7
  env TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config "$common" --ar-config "$ar_config" \
    --diffusion-config "$diff_config" \
    --diffusion-checkpoint "$diff_ckpt" --ar-checkpoint "$ar_ckpt" \
    --init-trained-checkpoint "$init_checkpoint" \
    --backbone-architecture "$architecture" \
    --backbone-initialization "$initialization" \
    --variant "$variant" --conditioner "$conditioner" \
    --pair-rank "$pair_rank" \
    --epochs 30 --patience 5 --min-epochs 5 \
    --batch-size 256 --eval-batch-size 64 \
    --backbone-lr 0.0001 --selector-lr 0.0005 \
    --weight-decay 0.0001 --token-loss-weight 0.1 \
    --proposal-k 72 --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    "$@" --output-dir "$root/$name" >"$root/$name.log" 2>&1
}

# Core selector ablation: can an independently scored unary model close the
# gap simply by training longer?
run_control unary_diffenc_converged \
  runs/parallel_drafter/video23_unary_diffenc_full_v1/best.pt \
  masked_decoder diffusion_pretrained unary diffusion_encoder 32

# DFlash-style conditioning control: frozen AR history features versus the
# drafter's own history encoder under a longer horizon.
run_control pairwise_arenc_converged \
  runs/parallel_drafter/video23_pairwise_arenc_full_v1/best.pt \
  masked_decoder diffusion_pretrained pairwise frozen_ar_encoder 32
