#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
out="$repo/runs/parameter_fairness/video23_full_20260830/independent_nonar_parammatch"
mkdir -p "$out"
cd "$repo"

/home/lingfengs111/miniconda3/envs/diffgrm/bin/python \
  scripts/train_parallel_path_verifier.py \
  --common-config experiments/canonical_full/video23_cf_official.yaml \
  --ar-config experiments/canonical_full/video23_ar_constrained.yaml \
  --diffusion-config experiments/canonical_full/diffusion_sequential.yaml \
  --diffusion-checkpoint saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin \
  --drafter-checkpoint runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt \
  --pair-rank 51 --proposal-k 72 --num-negatives 71 \
  --coordinate-mode bidirectional --history-pooling last \
  --hidden-dim 128 --coordinate-layers 2 --set-layers 2 \
  --independent-history-layers 3 \
  --independent-history-hidden-dim 256 \
  --independent-history-heads 4 \
  --independent-history-inner-dim 512 \
  --label-weight 1 --distill-weight 0 --margin-weight 0.1 --margin-value 0.2 \
  --epochs 6 --patience 2 --batch-size 64 --eval-batch-size 64 \
  --learning-rate 0.0003 --weight-decay 0.0001 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$out"

clean_repo=/home/lingfengs111/codes/GR/CleanGR
cd "$clean_repo"
/home/lingfengs111/.conda/envs/py313/bin/python -m cleangr.train.train_sasrec \
  --config configs/amazon23_video_game_sasrec_long_parammatch_6p7m.yaml
/home/lingfengs111/.conda/envs/py313/bin/python -m cleangr.evaluation.eval_sasrec \
  --config configs/amazon23_video_game_sasrec_long_parammatch_6p7m.yaml \
  --split test --ks 5,10,20,50 --mask-history

