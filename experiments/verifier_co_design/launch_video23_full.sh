#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
common=experiments/canonical_full/video23_cf_official.yaml
ar_config=experiments/canonical_full/video23_ar_constrained.yaml
diff_config=experiments/canonical_full/diffusion_sequential.yaml
diff_ckpt=saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin
ar_ckpt=saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin
drafter_ckpt=runs/next_round_20260828/video23_pairwise_rank51_continue2_v1/best.pt
root=runs/verifier_co_design/video23_full_20260829

cd "$repo"
mkdir -p "$root"

case "${1:?expected residual|token_control|parallel_then_joint}" in
  residual)
    out="$root/residual_ar_last_decoder"
    mkdir -p "$out"
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
      "$python_bin" scripts/train_candidate_aware_verifier.py \
      --dataset AmazonReviews2023CleanGR \
      --common-config "$common" --ar-config "$ar_config" \
      --diffusion-config "$diff_config" \
      --diffusion-checkpoint "$diff_ckpt" \
      --drafter-checkpoint "$drafter_ckpt" \
      --ar-checkpoint "$ar_ckpt" --pair-rank 51 \
      --epochs 3 --patience 1 --batch-size 32 --eval-batch-size 64 \
      --learning-rate 1e-5 --weight-decay 1e-4 \
      --token-loss-weight 0.1 --listwise-weight 1.0 \
      --margin-weight 0.1 --margin-value 0.2 \
      --training-fusion-alpha 0.75 --trainable-scope last_decoder \
      --num-negatives 15 --candidate-score-chunk-size 4 --proposal-k 72 \
      --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
      --output-dir "$out" > "$out/train.log" 2>&1
    ;;
  token_control)
    out="$root/token_ce_equal_budget"
    mkdir -p "$out"
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" PYTHONUNBUFFERED=1 \
      "$python_bin" scripts/train_candidate_aware_verifier.py \
      --dataset AmazonReviews2023CleanGR \
      --common-config "$common" --ar-config "$ar_config" \
      --diffusion-config "$diff_config" \
      --diffusion-checkpoint "$diff_ckpt" \
      --drafter-checkpoint "$drafter_ckpt" \
      --ar-checkpoint "$ar_ckpt" --pair-rank 51 \
      --epochs 3 --patience 1 --batch-size 32 --eval-batch-size 64 \
      --learning-rate 1e-5 --weight-decay 1e-4 \
      --token-loss-weight 1.0 --listwise-weight 0.0 \
      --margin-weight 0.0 --training-fusion-alpha 0.75 \
      --trainable-scope last_decoder \
      --num-negatives 15 --candidate-score-chunk-size 4 --proposal-k 72 \
      --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
      --output-dir "$out" > "$out/train.log" 2>&1
    ;;
  parallel_then_joint)
    stage2="$root/parallel_verifier_distill"
    stage3="$root/joint_recall_verify"
    mkdir -p "$stage2" "$stage3"
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" PYTHONUNBUFFERED=1 \
      "$python_bin" scripts/train_parallel_path_verifier.py \
      --dataset AmazonReviews2023CleanGR \
      --common-config "$common" --ar-config "$ar_config" \
      --diffusion-config "$diff_config" \
      --diffusion-checkpoint "$diff_ckpt" \
      --drafter-checkpoint "$drafter_ckpt" \
      --teacher-ar-checkpoint "$ar_ckpt" --pair-rank 51 \
      --epochs 4 --patience 2 --batch-size 64 --eval-batch-size 64 \
      --num-negatives 15 --proposal-k 72 --candidate-score-chunk-size 8 \
      --hidden-dim 128 --n-head 4 --coordinate-layers 1 --set-layers 1 \
      --learning-rate 3e-4 --label-weight 1.0 --distill-weight 0.5 \
      --distill-temperature 1.0 --margin-weight 0.1 --margin-value 0.2 \
      --fusion-alphas 0,0.25,0.5,0.75,1 \
      --output-dir "$stage2" > "$stage2/train.log" 2>&1

    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" PYTHONUNBUFFERED=1 \
      "$python_bin" scripts/train_parallel_path_verifier.py \
      --dataset AmazonReviews2023CleanGR \
      --common-config "$common" --ar-config "$ar_config" \
      --diffusion-config "$diff_config" \
      --diffusion-checkpoint "$diff_ckpt" \
      --drafter-checkpoint "$drafter_ckpt" \
      --teacher-ar-checkpoint "$ar_ckpt" \
      --init-verifier-checkpoint "$stage2/best.pt" --pair-rank 51 \
      --epochs 2 --patience 1 --batch-size 64 --eval-batch-size 64 \
      --num-negatives 15 --proposal-k 72 --candidate-score-chunk-size 8 \
      --hidden-dim 128 --n-head 4 --coordinate-layers 1 --set-layers 1 \
      --learning-rate 1e-4 --label-weight 1.0 --distill-weight 0.5 \
      --distill-temperature 1.0 --margin-weight 0.1 --margin-value 0.2 \
      --joint-drafter-weight 0.1 --drafter-learning-rate 1e-5 \
      --selector-learning-rate 1e-4 \
      --fusion-alphas 0,0.25,0.5,0.75,1 \
      --output-dir "$stage3" > "$stage3/train.log" 2>&1
    ;;
  *)
    echo "unknown experiment: $1" >&2
    exit 2
    ;;
esac

