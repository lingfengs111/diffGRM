#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/canonical_full/video23_multi_order_transfer"
gpu_id="${1:-3}"

common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
ar_config="$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml"
diff_config="$repo_dir/experiments/canonical_full/diffusion_sequential.yaml"
ar_ckpt="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin"
diff_ckpt="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin"
alphas="0,0.1,0.25,0.5,0.75,0.9,1"

mkdir -p "$output_dir"

run_eval() {
    local tag="$1"
    local split="$2"
    local orders="$3"
    env CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/evaluate_proposal_verifier.py" \
        --dataset=AmazonReviews2023CleanGR \
        --common-config="$common" \
        --ar-config="$ar_config" \
        --diffusion-config="$diff_config" \
        --ar-checkpoint="$ar_ckpt" \
        --diffusion-checkpoint="$diff_ckpt" \
        --proposal-mode=random \
        --proposal-k=32 \
        --decode-orders="$orders" \
        --output-k=10 \
        --metric-ks=5,10 \
        --split="$split" \
        --fusion-alphas="$alphas" \
        --batch-size=8 \
        --output="$output_dir/${tag}_${split}.json" \
        >"$output_dir/${tag}_${split}.log" 2>&1
}

one_order='0,1,2,3'
two_orders='0,1,2,3;3,2,1,0'
four_orders='0,1,2,3;2,3,0,1;3,2,1,0;1,0,3,2'

run_eval video23_one_order_k32 val "$one_order"
run_eval video23_one_order_k32 test "$one_order"
run_eval video23_two_orders_k32 val "$two_orders"
run_eval video23_two_orders_k32 test "$two_orders"
run_eval video23_four_orders_k32 val "$four_orders"
run_eval video23_four_orders_k32 test "$four_orders"
