#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
ar_config="$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml"
diff_config="$repo_dir/experiments/canonical_full/diffusion_sequential.yaml"
sid_config="$repo_dir/experiments/rq_opq/video23_rq2_opq2_cf.yaml"
run_root="$repo_dir/runs/next_round_20260828"
mkdir -p "$run_root"

env CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/main.py" \
    --model AR_GRM --dataset AmazonReviews2023CleanGR \
    --config "$common" --config "$sid_config" --config "$ar_config" \
    --run_id=video23_full_rq2opq2_cf_ar_v1 \
    >"$run_root/video23_full_rq2opq2_cf_ar_v1.log" 2>&1 &
rq_ar_pid=$!
echo "$rq_ar_pid" >"$run_root/rq_ar.pid"

env CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/main.py" \
    --model DIFF_GRM --dataset AmazonReviews2023CleanGR \
    --config "$common" --config "$sid_config" \
    --run_id=video23_full_rq2opq2_cf_diff_guided_v1 \
    >"$run_root/video23_full_rq2opq2_cf_diff_guided_v1.log" 2>&1 &
rq_diff_pid=$!
echo "$rq_diff_pid" >"$run_root/rq_diff.pid"

(
    while kill -0 "$rq_ar_pid" 2>/dev/null || kill -0 "$rq_diff_pid" 2>/dev/null; do
        sleep 30
    done
    rq_ar_checkpoint="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_rq2opq2_cf_ar_v1/pytorch_model.bin"
    rq_diff_checkpoint="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_rq2opq2_cf_diff_guided_v1/pytorch_model.bin"
    if [[ ! -s "$rq_ar_checkpoint" || ! -s "$rq_diff_checkpoint" ]]; then
        echo "RQ-OPQ backbone training ended without both checkpoints" >&2
        exit 1
    fi
    env CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/train_parallel_opq_drafter.py" \
        --dataset AmazonReviews2023CleanGR \
        --common-config "$common" --sid-config "$sid_config" \
        --ar-config "$ar_config" --diffusion-config "$diff_config" \
        --diffusion-checkpoint "$rq_diff_checkpoint" \
        --ar-checkpoint "$rq_ar_checkpoint" \
        --variant pairwise --conditioner diffusion_encoder --pair-rank 32 \
        --epochs 5 --batch-size 256 --eval-batch-size 64 \
        --backbone-lr 0.0001 --selector-lr 0.001 \
        --token-loss-weight 0.1 --proposal-k 72 \
        --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
        --output-dir "$run_root/video23_rq2opq2_pairwise_r32_v1" \
        >"$run_root/video23_rq2opq2_pairwise_r32_v1.log" 2>&1
) &
rq_pipeline_pid=$!
echo "$rq_pipeline_pid" >"$run_root/rq_pipeline.pid"
echo "rq_ar=$rq_ar_pid rq_diff=$rq_diff_pid rq_pipeline=$rq_pipeline_pid"
wait "$rq_pipeline_pid"
