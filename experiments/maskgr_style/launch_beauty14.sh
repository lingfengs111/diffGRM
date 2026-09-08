#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/canonical_full/maskgr_style"
common_config="$repo_dir/experiments/canonical_full/beauty14_full.yaml"
mkdir -p "$output_dir"

launch_train() {
    local gpu_id="$1"
    local variant_config="$2"
    local run_id="$3"
    env CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/main.py" \
        --model=DIFF_GRM \
        --dataset=AmazonReviews2014CleanGR \
        --config="$common_config" \
        --config="$repo_dir/$variant_config" \
        --run_id="$run_id" \
        >"$output_dir/$run_id.log" 2>&1 &
    echo "$!" >"$output_dir/$run_id.pid"
    echo "launched gpu=$gpu_id pid=$! run_id=$run_id"
}

launch_train 1 \
    experiments/maskgr_style/beauty14_target_uniform.yaml \
    beauty14_full_opq_maskgr_target_uniform_v1

launch_train 2 \
    experiments/maskgr_style/beauty14_target_history_uniform.yaml \
    beauty14_full_opq_maskgr_target_history_uniform_w0p2_v1

wait
