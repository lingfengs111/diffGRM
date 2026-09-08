#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_root/runs/motivation_2x2"
mkdir -p "$output_dir"

launch_one() {
    local gpu="$1"
    local sid_config="$2"
    local run_id="$3"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$repo_root/main.py" \
        --model=AR_GRM \
        --dataset=AmazonReviews2014 \
        --config="$repo_root/experiments/motivation_2x2/common.yaml" \
        --config="$repo_root/experiments/motivation_2x2/$sid_config" \
        --config="$repo_root/experiments/motivation_2x2/ar_alternating.yaml" \
        --run_id="$run_id" \
        >"$output_dir/$run_id.log" 2>&1 &
    echo "$!" >"$output_dir/$run_id.pid"
}

launch_one 1 opq_decaware_sem0p1.yaml beauty14_opq_decaware_sem0p1_ar_alt1
launch_one 2 opq_decaware_sem0p03.yaml beauty14_opq_decaware_sem0p03_ar_alt1
launch_one 3 opq_decaware_sem0.yaml beauty14_opq_decaware_sem0_ar_alt1
wait
