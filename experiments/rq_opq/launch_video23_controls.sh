#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
common="$repo_dir/experiments/canonical_full/video23_cf_official.yaml"
ar_config="$repo_dir/experiments/canonical_full/video23_ar_constrained.yaml"
diff_config="$repo_dir/experiments/canonical_full/diffusion_sequential.yaml"
ar_checkpoint="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_ar_v1/pytorch_model.bin"
diff_checkpoint="$repo_dir/saved/AmazonReviews2023CleanGR_video23_full_opq_cf_diff_guided_v1/pytorch_model.bin"
pair_checkpoint="$repo_dir/runs/parallel_drafter/video23_pairwise_diffenc_full_v1/best.pt"
mips_checkpoint="$repo_dir/runs/ann_drafter/video23_exact_mips_id_v1/best.pt"
run_root="$repo_dir/runs/next_round_20260828"
mkdir -p "$run_root"

(
    env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/train_ann_drafter.py" \
        --dataset AmazonReviews2023CleanGR \
        --common-config "$common" --ar-config "$ar_config" \
        --ar-checkpoint "$ar_checkpoint" --item-mode id \
        --init-trained-checkpoint "$mips_checkpoint" --epochs 0 \
        --batch-size 512 --eval-batch-size 64 --proposal-k 32 \
        --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
        --output-dir "$run_root/video23_exact_mips_id_k32" \
        >"$run_root/video23_exact_mips_id_k32.log" 2>&1
    env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/evaluate_drafter_overlap.py" \
        --dataset AmazonReviews2023CleanGR \
        --common-config "$common" --ar-config "$ar_config" \
        --diffusion-config "$diff_config" \
        --ar-checkpoint "$ar_checkpoint" \
        --diffusion-checkpoint "$diff_checkpoint" \
        --mips-checkpoint "$mips_checkpoint" \
        --semantic-checkpoint "$pair_checkpoint" \
        --proposal-k 32 --eval-batch-size 64 \
        --output "$run_root/video23_mips_pairwise_k32_overlap.json" \
        >"$run_root/video23_mips_pairwise_k32_overlap.log" 2>&1
) &
control_pid=$!

env CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
    "$python_bin" "$repo_dir/scripts/train_parallel_opq_drafter.py" \
    --dataset AmazonReviews2023CleanGR \
    --common-config "$common" --ar-config "$ar_config" \
    --diffusion-config "$diff_config" \
    --diffusion-checkpoint "$diff_checkpoint" \
    --ar-checkpoint "$ar_checkpoint" \
    --init-trained-checkpoint "$pair_checkpoint" \
    --variant triple --conditioner diffusion_encoder \
    --pair-rank 32 --triple-rank 16 --epochs 2 \
    --batch-size 256 --eval-batch-size 64 \
    --backbone-lr 0.0001 --selector-lr 0.001 \
    --token-loss-weight 0.1 --proposal-k 72 \
    --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
    --output-dir "$run_root/video23_pairwise_triple_r16_v1" \
    >"$run_root/video23_pairwise_triple_r16_v1.log" 2>&1 &
triple_pid=$!

echo "$control_pid" >"$run_root/mips_overlap.pid"
echo "$triple_pid" >"$run_root/triple.pid"
wait "$control_pid" "$triple_pid"

