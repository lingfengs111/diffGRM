#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
common="$repo/experiments/canonical_full/video23_cf_official.yaml"
ar_config="$repo/experiments/canonical_full/video23_ar_constrained.yaml"
diff_config="$repo/experiments/canonical_full/diffusion_sequential.yaml"
diff_ckpt="$repo/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin"
drafter_ckpt="$repo/runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt"
root="$repo/runs/verifier_only/video23_full_20260830"
mode="${1:?expected bidirectional|causal|mlp}"
gpu="${2:?expected physical GPU index}"

case "$mode" in
  bidirectional|causal|mlp) ;;
  *) echo "unknown coordinate mode: $mode" >&2; exit 2 ;;
esac

out="$root/${mode}_last_k72_random"
mkdir -p "$out"
cd "$repo"

echo "$(date -Is) mode=$mode gpu=$gpu candidates=72 history_pooling=last" \
  | tee -a "$out/launcher.log"
exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$python_bin" scripts/train_parallel_path_verifier.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config "$common" --ar-config "$ar_config" \
  --diffusion-config "$diff_config" \
  --diffusion-checkpoint "$diff_ckpt" \
  --drafter-checkpoint "$drafter_ckpt" --pair-rank 51 \
  --epochs 6 --patience 2 --batch-size 64 --eval-batch-size 64 \
  --num-negatives 71 --proposal-k 72 \
  --hidden-dim 128 --n-head 4 --coordinate-layers 1 --set-layers 1 \
  --coordinate-mode "$mode" --history-pooling last \
  --learning-rate 3e-4 --weight-decay 1e-4 \
  --label-weight 1.0 --distill-weight 0.0 \
  --margin-weight 0.1 --margin-value 0.2 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --seed 2026 --output-dir "$out" > "$out/train.log" 2>&1
