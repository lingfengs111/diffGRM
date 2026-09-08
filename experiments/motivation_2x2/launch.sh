#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/motivation_2x2"
mkdir -p "$output_dir"

launch_one() {
    local gpu="$1"
    local model="$2"
    local sid_config="$3"
    local model_config="$4"
    local run_id="$5"
    local log_path="$output_dir/$run_id.log"
    env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
        nohup "$python_bin" "$repo_dir/main.py" \
        --model="$model" \
        --dataset=AmazonReviews2014 \
        --config="$repo_dir/experiments/motivation_2x2/common.yaml" \
        --config="$repo_dir/experiments/motivation_2x2/$sid_config" \
        --config="$repo_dir/experiments/motivation_2x2/$model_config" \
        --run_id="$run_id" \
        >"$log_path" 2>&1 &
    echo "$!" >"$output_dir/$run_id.pid"
    echo "launched gpu=$gpu pid=$! run=$run_id log=$log_path"
}

cd "$repo_dir"
launch_one 0 AR_GRM rq_cf.yaml ar.yaml beauty14_rq_cf_ar_v1
launch_one 1 DIFF_GRM rq_cf.yaml diffusion.yaml beauty14_rq_cf_diff_v1
launch_one 2 AR_GRM opq_cf.yaml ar.yaml beauty14_opq_cf_ar_v1
launch_one 3 DIFF_GRM opq_cf.yaml diffusion.yaml beauty14_opq_cf_diff_v1

# Keep a supervising parent alive.  Some remote execution environments reap
# detached descendants when the launcher exits even if nohup is used.
wait
