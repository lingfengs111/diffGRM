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
mkdir -p "$output_dir"

while true; do
    used=$(nvidia-smi -i 0 --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 300 )); then
        sleep 20
        used=$(nvidia-smi -i 0 --query-gpu=memory.used \
            --format=csv,noheader,nounits | tr -d ' ')
        [[ "$used" =~ ^[0-9]+$ ]] && (( used < 300 )) && break
    fi
    sleep 30
done

four_orders='0,1,2,3;2,3,0,1;3,2,1,0;1,0,3,2'
two_orders='0,1,2,3;3,2,1,0'
one_order='0,1,2,3'

env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/benchmark_inference_latency.py" \
    --dataset=AmazonReviews2014CleanGR \
    --common-config="$common" \
    --ar-config="$ar_config" \
    --diffusion-config="$diff_config" \
    --ar-checkpoint="$ar_ckpt" \
    --diffusion-checkpoint="$diff_ckpt" \
    --decode-orders="$four_orders" \
    --proposal-k=32 --standalone-beam=128 --batch-size=8 \
    --max-examples=512 --warmup-batches=2 --fusion-alpha=0.5 \
    --output="$output_dir/beauty14_cached_encoder_b8_n512.json" \
    >"$output_dir/beauty14_cached_encoder_b8_n512.log" 2>&1

run_val() {
    local tag="$1"
    local orders="$2"
    env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/evaluate_proposal_verifier.py" \
        --dataset=AmazonReviews2014CleanGR \
        --common-config="$common" \
        --ar-config="$ar_config" \
        --diffusion-config="$diff_config" \
        --ar-checkpoint="$ar_ckpt" \
        --diffusion-checkpoint="$diff_ckpt" \
        --proposal-mode=random --proposal-k=32 \
        --decode-orders="$orders" \
        --output-k=10 --metric-ks=5,10 --split=val \
        --fusion-alphas=0.25,0.5,0.75 --batch-size=8 \
        --output="$output_dir/$tag.json" \
        >"$output_dir/$tag.log" 2>&1
}

run_val beauty14_one_order_k32_fusion_val "$one_order"
run_val beauty14_two_orders_k32_fusion_val "$two_orders"
