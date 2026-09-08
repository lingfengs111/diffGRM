#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_root="$repo/runs/capacity_fairness/video23_half_2x2_d176/candidate_budget"
trained="$repo/runs/capacity_fairness/video23_half_2x2_d176/random_encoder4_pairwise_half_ar/best.pt"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_half_2x2_d176_v1/pytorch_model.bin"
mkdir -p "$run_root"
cd "$repo"
test -s "$trained"
test -s "$ar_ckpt"

# Evaluation-only sweep: identical trained drafter/verifier, changing only
# how many proposals the AR verifier may inspect.  K=72 reproduces the legacy
# Beauty14 four-order-union budget; round K values expose the trade-off.
for proposal_k in 10 32 64 72 128; do
  output_dir="$run_root/k${proposal_k}"
  if [[ -s "$output_dir/result.json" ]]; then
    continue
  fi
  env TOKENIZERS_PARALLELISM=false "$python_bin" \
    scripts/train_parallel_opq_drafter.py \
    --dataset AmazonReviews2023CleanGR \
    --common-config experiments/canonical_full/video23_cf_official.yaml \
    --sid-config experiments/capacity_fairness/video23_half_2x2_d176.yaml \
    --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
    --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
    --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_half_2x2_d176_v1/pytorch_model.bin \
    --ar-checkpoint "$ar_ckpt" \
    --backbone-architecture encoder_four_head \
    --backbone-initialization random \
    --encoder-head-n-layer 4 \
    --variant pairwise --conditioner diffusion_encoder --pair-rank 35 \
    --training-objective catalog_plus_token \
    --epochs 0 --init-trained-checkpoint "$trained" \
    --batch-size 256 --eval-batch-size 64 \
    --proposal-k "$proposal_k" --seed 2026 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --output-dir "$output_dir" \
    >"$run_root/k${proposal_k}.log" 2>&1
done
