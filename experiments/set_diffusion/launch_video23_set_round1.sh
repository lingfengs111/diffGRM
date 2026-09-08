#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
ar_config="$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml"
diff_config="$repo_dir/experiments/canonical_full/diffusion_sequential.yaml"
diff_checkpoint="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin"
ar_checkpoint="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin"
pair_checkpoint="$repo_dir/runs/next_round_20260828/video23_pairwise_rank51_continue2_v1/best.pt"
run_root="$repo_dir/runs/set_diffusion/round1_video23"
mkdir -p "$run_root"

common_args=(
    --dataset AmazonReviews2023CleanGR
    --common-config "$common"
    --ar-config "$ar_config"
    --diffusion-config "$diff_config"
    --diffusion-checkpoint "$diff_checkpoint"
    --ar-checkpoint "$ar_checkpoint"
    --init-trained-checkpoint "$pair_checkpoint"
    --variant pairwise --conditioner diffusion_encoder --pair-rank 51
    --batch-size 128 --eval-batch-size 16
    --backbone-lr 0.0001 --selector-lr 0.001
    --token-loss-weight 0.1 --subset-mask-views 1
    --proposal-k 72 --two-pass-branches 16
    --two-pass-first-weights 0,0.25,0.5,0.75,1
    --two-pass-branch-chunk 16 --skip-ar-verifier
)

# A) Uniform coverage of every non-full typed subset (mask widths 1--3).
nohup env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/train_parallel_opq_drafter.py" \
    "${common_args[@]}" \
    --epochs 2 --subset-loss-weight 0.5 \
    --subset-min-masked 1 --subset-max-masked 3 \
    --output-dir "$run_root/set_all_w0p5" \
    >"$run_root/set_all_w0p5.log" 2>&1 &
all_pid=$!
echo "$all_pid" >"$run_root/set_all_w0p5.pid"

# B) Harder conditional views: at least half of the SID remains unknown.
nohup env CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/train_parallel_opq_drafter.py" \
    "${common_args[@]}" \
    --epochs 2 --subset-loss-weight 1.0 \
    --subset-min-masked 2 --subset-max-masked 3 \
    --output-dir "$run_root/set_hard_w1" \
    >"$run_root/set_hard_w1.log" 2>&1 &
hard_pid=$!
echo "$hard_pid" >"$run_root/set_hard_w1.pid"

# C) No-training control: does extra route breadth alone rescue the old model?
nohup env CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/train_parallel_opq_drafter.py" \
    "${common_args[@]}" \
    --epochs 0 --subset-loss-weight 0 \
    --output-dir "$run_root/old_model_two_pass_b16" \
    >"$run_root/old_model_two_pass_b16.log" 2>&1 &
control_pid=$!
echo "$control_pid" >"$run_root/old_model_two_pass_b16.pid"

echo "set_all=$all_pid set_hard=$hard_pid old_two_pass=$control_pid"

if [[ "${WAIT_FOR_RUNS:-0}" == "1" ]]; then
    wait "$all_pid" "$hard_pid" "$control_pid"
fi
