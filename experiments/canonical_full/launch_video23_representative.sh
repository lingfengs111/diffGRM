#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/canonical_full/video23_representative"
common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
mkdir -p "$output_dir"

wait_for_gpu() {
    local gpu_id="$1"
    while true; do
        local used
        used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used \
            --format=csv,noheader,nounits | tr -d ' ')
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 300 )); then
            # Require a second clean observation to avoid racing a job that is
            # between data preparation and CUDA initialization.
            sleep 20
            used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used \
                --format=csv,noheader,nounits | tr -d ' ')
            if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 300 )); then
                return
            fi
        fi
        sleep 30
    done
}

launch_after_free() {
    local gpu_id="$1"
    local model="$2"
    local run_id="$3"
    shift 3
    wait_for_gpu "$gpu_id"
    local config_args=()
    for config_path in "$@"; do
        config_args+=(--config="$repo_dir/$config_path")
    done
    echo "$(date -Is) launching gpu=$gpu_id run_id=$run_id" \
        >>"$output_dir/launcher.log"
    env CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/main.py" \
        --model="$model" \
        --dataset=AmazonReviews2023CleanGR \
        --config="$common" \
        "${config_args[@]}" \
        --run_id="$run_id" \
        >"$output_dir/$run_id.log" 2>&1
}

# GPU 2 becomes available after the currently running official Video23 test.
launch_after_free 2 AR_GRM video23_full_opq_cf_ar_v1 \
    experiments/canonical_full/video23_ar_constrained.yaml &

# These two GPUs currently host unrelated jobs, so wait rather than preempt.
launch_after_free 1 DIFF_GRM video23_full_opq_cf_diff_sequential_v1 \
    experiments/canonical_full/diffusion_sequential.yaml &

launch_after_free 3 DIFF_GRM video23_full_opq_cf_diff_seq_history_w0p2_v1 \
    experiments/canonical_full/diffusion_sequential.yaml \
    experiments/canonical_full/video23_sequential_history.yaml &

wait
