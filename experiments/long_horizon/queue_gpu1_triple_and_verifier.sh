#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/long_horizon_20260831"
while tmux has-session -t video23_full_converge_watch_0831 2>/dev/null; do
  sleep 30
done
mkdir -p "$root"
cd "$repo"

# Add the triple residual only after the rank-51 pairwise model has converged;
# the old triple run used an earlier rank-32 checkpoint and only two epochs.
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin \
  --init-trained-checkpoint runs/capacity_fairness/video23_continuations_20260831/one_pass_pairwise_converged/best.pt \
  --backbone-architecture masked_decoder \
  --backbone-initialization diffusion_pretrained \
  --variant triple --conditioner diffusion_encoder \
  --pair-rank 51 --triple-rank 16 \
  --epochs 30 --patience 5 --min-epochs 5 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.00005 --selector-lr 0.0005 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$root/triple_r16_from_converged_pairwise" \
  >"$root/triple_r16_from_converged_pairwise.log" 2>&1
