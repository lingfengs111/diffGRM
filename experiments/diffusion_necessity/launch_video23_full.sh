#!/usr/bin/env bash
set -euo pipefail

repo="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
run_root="$repo/runs/diffusion_necessity/video23_full_20260829"
common="$repo/experiments/canonical_full/video23_cf_official.yaml"
ar_config="$repo/experiments/canonical_full/video23_ar_constrained.yaml"
diff_config="$repo/experiments/canonical_full/diffusion_sequential.yaml"
diff_ckpt="$repo/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin"
mkdir -p "$run_root"

run_arm() {
    local name="$1"
    local architecture="$2"
    local initialization="$3"
    echo "$(date -Is) starting $name" | tee -a "$run_root/launcher.log"
    env CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo/scripts/train_parallel_opq_drafter.py" \
        --dataset AmazonReviews2023CleanGR \
        --common-config "$common" \
        --ar-config "$ar_config" \
        --diffusion-config "$diff_config" \
        --diffusion-checkpoint "$diff_ckpt" \
        --ar-checkpoint "$ar_ckpt" \
        --backbone-architecture "$architecture" \
        --backbone-initialization "$initialization" \
        --encoder-head-n-layer 4 \
        --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
        --epochs 10 --batch-size 256 --eval-batch-size 64 \
        --backbone-lr 0.0003 --selector-lr 0.001 \
        --weight-decay 0.0001 --token-loss-weight 0.1 \
        --proposal-k 72 --seed 2026 \
        --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
        --output-dir "$run_root/$name" \
        >"$run_root/$name.log" 2>&1
    echo "$(date -Is) completed $name" | tee -a "$run_root/launcher.log"
}

# A-B isolates the value of DiffGRM denoising pretraining.
run_arm diff_pretrained_pairwise_r51 \
    masked_decoder diffusion_pretrained
run_arm masked_random_pairwise_r51 \
    masked_decoder random

# B-C isolates the masked decoder against a parameter-matched encoder-only
# four-head model; both are trained from random initialization.
run_arm encoder4_four_head_pairwise_r51 \
    encoder_four_head random

