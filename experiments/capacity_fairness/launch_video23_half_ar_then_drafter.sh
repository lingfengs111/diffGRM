#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_root="$repo/runs/capacity_fairness/video23_half_2x2_d176"
ar_id=video23_full_opq_cf_ar_half_2x2_d176_v1
diff_id=video23_full_opq_cf_diff_guided_half_2x2_d176_v1
mkdir -p "$run_root"
cd "$repo"

env TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
  --model=AR_GRM \
  --dataset=AmazonReviews2023CleanGR \
  --config=experiments/canonical_full/video23_cf_official.yaml \
  --config=experiments/canonical_full/video23_ar_constrained.yaml \
  --config=experiments/capacity_fairness/video23_half_2x2_d176.yaml \
  --run_id="$ar_id" \
  >"$run_root/$ar_id.log" 2>&1

# The final one-pass model must use the fully trained small checkpoints, not an
# intermediate checkpoint emitted at the first validation epoch.
while tmux has-session -t video23_half_diff_0831 2>/dev/null; do
  sleep 30
done

diff_ckpt="$repo/saved/AmazonReviews2023CleanGR_${diff_id}/pytorch_model.bin"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_id}/pytorch_model.bin"
test -s "$diff_ckpt"
test -s "$ar_ckpt"

env TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --sid-config experiments/capacity_fairness/video23_half_2x2_d176.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint "$diff_ckpt" \
  --ar-checkpoint "$ar_ckpt" \
  --backbone-architecture masked_decoder \
  --backbone-initialization diffusion_pretrained \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 35 \
  --epochs 30 --patience 5 --min-epochs 10 \
  --batch-size 256 --eval-batch-size 64 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --output-dir "$run_root/one_pass_pairwise_ar_fusion" \
  >"$run_root/one_pass_pairwise_ar_fusion.log" 2>&1
