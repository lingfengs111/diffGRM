#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/motivation_2x2/controls"
mkdir -p "$output_dir"

run_control() {
    local gpu="$1"
    local sid_name="$2"
    local sid_config="$3"
    local ar_run="$4"
    local diff_run="$5"
    local proposal_k="$6"
    local decode_orders="$7"
    local tag="$8"

    local result_path="$output_dir/${sid_name}_${tag}.json"
    local log_path="$output_dir/${sid_name}_${tag}.log"

    env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/evaluate_proposal_verifier.py" \
        --dataset=AmazonReviews2014 \
        --common-config="$repo_dir/experiments/motivation_2x2/common.yaml" \
        --sid-config="$repo_dir/experiments/motivation_2x2/$sid_config" \
        --ar-config="$repo_dir/experiments/motivation_2x2/ar.yaml" \
        --diffusion-config="$repo_dir/experiments/motivation_2x2/diffusion.yaml" \
        --ar-checkpoint="$repo_dir/saved/AmazonReviews2014_${ar_run}/pytorch_model.bin" \
        --diffusion-checkpoint="$repo_dir/saved/AmazonReviews2014_${diff_run}/pytorch_model.bin" \
        --proposal-k="$proposal_k" \
        --output-k=10 \
        --batch-size=8 \
        --decode-orders="$decode_orders" \
        --output="$result_path" \
        >"$log_path" 2>&1
}

run_control 0 opq opq_cf.yaml \
    beauty14_opq_cf_ar_constrained_v2 beauty14_opq_cf_diff_v1 \
    128 '0,1,2,3' single_order_k128 &
opq_single_pid=$!
echo "$opq_single_pid" >"$output_dir/opq_single_order_k128.pid"

run_control 1 rq rq_cf.yaml \
    beauty14_rq_cf_ar_constrained_v2 beauty14_rq_cf_diff_v1 \
    128 '0,1,2,3' single_order_k128 &
rq_single_pid=$!
echo "$rq_single_pid" >"$output_dir/rq_single_order_k128.pid"

run_control 2 opq opq_cf.yaml \
    beauty14_opq_cf_ar_constrained_v2 beauty14_opq_cf_diff_v1 \
    32 '0,1,2,3;2,3,0,1;3,2,1,0;1,0,3,2' multi_order_k32 &
opq_multi_pid=$!
echo "$opq_multi_pid" >"$output_dir/opq_multi_order_k32.pid"

run_control 3 rq rq_cf.yaml \
    beauty14_rq_cf_ar_constrained_v2 beauty14_rq_cf_diff_v1 \
    32 '0,1,2,3;2,3,0,1;3,2,1,0;1,0,3,2' multi_order_k32 &
rq_multi_pid=$!
echo "$rq_multi_pid" >"$output_dir/rq_multi_order_k32.pid"

echo "controls launched: opq_single=$opq_single_pid rq_single=$rq_single_pid opq_multi=$opq_multi_pid rq_multi=$rq_multi_pid"
wait "$opq_single_pid" "$rq_single_pid" "$opq_multi_pid" "$rq_multi_pid"
