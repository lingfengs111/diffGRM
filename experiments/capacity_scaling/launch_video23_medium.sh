#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/capacity_scaling/video23_medium_20260829"
common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
medium="$repo_dir/experiments/capacity_scaling/video23_medium_3x3_d256_ffn1024.yaml"
mkdir -p "$output_dir"

launch_one() {
    local gpu_id="$1"
    local model="$2"
    local run_id="$3"
    shift 3

    local config_args=()
    for config_path in "$@"; do
        config_args+=(--config="$config_path")
    done

    echo "$(date -Is) launching gpu=$gpu_id model=$model run_id=$run_id" \
        | tee -a "$output_dir/launcher.log"
    env CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/main.py" \
        --model="$model" \
        --dataset=AmazonReviews2023CleanGR \
        --config="$common" \
        "${config_args[@]}" \
        --config="$medium" \
        --run_id="$run_id" \
        >"$output_dir/$run_id.log" 2>&1 &
    echo "$!" >"$output_dir/$run_id.pid"
}

launch_one 0 AR_GRM video23_full_opq_cf_ar_medium_3x3_v1 \
    "$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml"

launch_one 1 DIFF_GRM video23_full_opq_cf_diff_guided_medium_3x3_v1

wait
