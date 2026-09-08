#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/canonical_full/speed_tradeoff"
common="$repo_dir/experiments/canonical_full/beauty14_full.yaml"
ar_config="$repo_dir/experiments/canonical_full/ar_constrained.yaml"
diff_config="$repo_dir/experiments/canonical_full/diffusion_sequential.yaml"
ar_ckpt="$repo_dir/saved/AmazonReviews2014CleanGR_beauty14_full_opq_ar_finetune_control_v1/pytorch_model.bin"
diff_ckpt="$repo_dir/saved/AmazonReviews2014CleanGR_beauty14_full_opq_cf_diff_v1/pytorch_model.bin"
two_orders='0,1,2,3;3,2,1,0'

mkdir -p "$output_dir"

env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/evaluate_proposal_verifier.py" \
    --dataset=AmazonReviews2014CleanGR \
    --common-config="$common" \
    --ar-config="$ar_config" \
    --diffusion-config="$diff_config" \
    --ar-checkpoint="$ar_ckpt" \
    --diffusion-checkpoint="$diff_ckpt" \
    --proposal-mode=random --proposal-k=32 \
    --decode-orders="$two_orders" \
    --output-k=10 --metric-ks=5,10 --split=test \
    --fusion-alphas=0.75 --batch-size=8 \
    --output="$output_dir/beauty14_two_orders_k32_fusion_a0p75_test.json" \
    >"$output_dir/beauty14_two_orders_k32_fusion_a0p75_test.log" 2>&1

env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/benchmark_inference_latency.py" \
    --dataset=AmazonReviews2014CleanGR \
    --common-config="$common" \
    --ar-config="$ar_config" \
    --diffusion-config="$diff_config" \
    --ar-checkpoint="$ar_ckpt" \
    --diffusion-checkpoint="$diff_ckpt" \
    --decode-orders="$two_orders" \
    --proposal-k=32 --standalone-beam=128 --batch-size=8 \
    --max-examples=512 --warmup-batches=2 --fusion-alpha=0.75 \
    --output="$output_dir/beauty14_two_orders_cached_encoder_b8_n512.json" \
    >"$output_dir/beauty14_two_orders_cached_encoder_b8_n512.log" 2>&1
