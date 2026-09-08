#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
sid="$repo_dir/experiments/rq_opq/video23_rq2_opq2_cf.yaml"
ar="$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml"
run_root="$repo_dir/runs/next_round_20260828"

# The test jobs currently own GPUs 0 and 1.  Their JSON outputs are atomic
# completion markers; continuation starts only after each corresponding test.
(
    while [[ ! -s "$run_root/video23_rq2opq2_ar_test.json" ]]; do sleep 20; done
    env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/main.py" \
        --model AR_GRM --dataset AmazonReviews2023CleanGR \
        --config "$common" --config "$sid" --config "$ar" \
        --config "$repo_dir/experiments/rq_opq/video23_rq2opq2_ar_continue.yaml" \
        --run_id=video23_full_rq2opq2_cf_ar_continue_v1 \
        >"$run_root/video23_full_rq2opq2_cf_ar_continue_v1.log" 2>&1
) &
ar_pid=$!

(
    while [[ ! -s "$run_root/video23_rq2opq2_diff_test.json" ]]; do sleep 20; done
    env CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/main.py" \
        --model DIFF_GRM --dataset AmazonReviews2023CleanGR \
        --config "$common" --config "$sid" \
        --config "$repo_dir/experiments/rq_opq/video23_rq2opq2_diff_continue.yaml" \
        --run_id=video23_full_rq2opq2_cf_diff_continue_v1 \
        >"$run_root/video23_full_rq2opq2_cf_diff_continue_v1.log" 2>&1
) &
diff_pid=$!

echo "rqopq_continue_ar=$ar_pid rqopq_continue_diff=$diff_pid"
wait "$ar_pid" "$diff_pid"
