#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
root="$repo/runs/long_horizon_20260831"
while tmux has-session -t video23_large_diffgrm_0830 2>/dev/null; do
  sleep 30
done
mkdir -p "$root"
cd "$repo"

# The fair RQ2+OPQ2 arm uses its fully continued direct checkpoints, but the
# structured drafter itself previously received only two continuation epochs.
env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --sid-config experiments/rq_opq/video23_rq2_opq2_cf.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_rq2opq2_cf_diff_continue_v1/pytorch_model.bin \
  --ar-checkpoint saved/AmazonReviews2023CleanGR_video23_full_rq2opq2_cf_ar_continue_v1/pytorch_model.bin \
  --init-trained-checkpoint runs/next_round_20260828/video23_rq2opq2_pairwise_strong_ar_continue2_v1/best.pt \
  --backbone-architecture masked_decoder \
  --backbone-initialization diffusion_pretrained \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 32 \
  --epochs 30 --patience 5 --min-epochs 5 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0001 --selector-lr 0.0005 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$root/rq2opq2_pairwise_converged" \
  >"$root/rq2opq2_pairwise_converged.log" 2>&1

# The old canonical direct DiffGRM job ended after epoch 54 without satisfying
# its patience rule.  Re-run it to a genuine early stop for a paper-grade
# direct-generation baseline.
env TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=DIFF_GRM --dataset=AmazonReviews2023CleanGR \
  --config=experiments/canonical_full/video23_cf_official.yaml \
  --run_id=video23_full_opq_cf_diff_guided_complete_v2 \
  >"$root/video23_full_opq_cf_diff_guided_complete_v2.log" 2>&1

