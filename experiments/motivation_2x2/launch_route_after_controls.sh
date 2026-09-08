#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
run_id="beauty14_opq_behavior_route64_ar_v1"
log_path="$repo_dir/runs/motivation_2x2/$run_id.log"

while pgrep -f \
    'evaluate_proposal_verifier.py.*controls/opq_single_order_k128.json' \
    >/dev/null; do
    sleep 30
done

cd "$repo_dir"
env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/main.py" \
    --model=AR_GRM \
    --dataset=AmazonReviews2014 \
    --config="$repo_dir/experiments/motivation_2x2/common.yaml" \
    --config="$repo_dir/experiments/motivation_2x2/opq_behavior_route64.yaml" \
    --config="$repo_dir/experiments/motivation_2x2/ar_route.yaml" \
    --run_id="$run_id" \
    >"$log_path" 2>&1 &
route_pid=$!
echo "$route_pid" >"$repo_dir/runs/motivation_2x2/$run_id.pid"
echo "route launched: gpu=0 pid=$route_pid run=$run_id"
wait "$route_pid"
