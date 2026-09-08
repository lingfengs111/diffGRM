#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_root/runs/canonical_full"
mkdir -p "$output_dir"

launch_one() {
    local gpu="$1"
    local model="$2"
    local dataset="$3"
    local run_id="$4"
    shift 4
    local config_args=()
    for config_path in "$@"; do
        config_args+=(--config="$repo_root/$config_path")
    done
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$repo_root/main.py" \
        --model="$model" \
        --dataset="$dataset" \
        "${config_args[@]}" \
        --run_id="$run_id" \
        >"$output_dir/$run_id.log" 2>&1 &
    echo "$!" >"$output_dir/$run_id.pid"
    echo "launched gpu=$gpu pid=$! run_id=$run_id"
}

launch_one 1 DIFF_GRM AmazonReviews2014CleanGR \
    beauty14_full_opq_cf_diff_v1 \
    experiments/canonical_full/beauty14_full.yaml \
    experiments/canonical_full/diffusion_sequential.yaml

launch_one 2 AR_GRM AmazonReviews2014CleanGR \
    beauty14_full_opq_cf_ar_v1 \
    experiments/canonical_full/beauty14_full.yaml \
    experiments/canonical_full/ar_constrained.yaml

launch_one 3 DIFF_GRM AmazonReviews2023CleanGR \
    video23_full_opq_cf_diff_guided_v1 \
    experiments/canonical_full/video23_cf_official.yaml

wait
