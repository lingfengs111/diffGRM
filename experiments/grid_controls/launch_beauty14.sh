#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
run_dir="$repo_dir/runs/canonical_full/grid_controls"
mkdir -p "$run_dir"

nohup env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  "$python_bin" "$repo_dir/scripts/evaluate_checkpoint.py" \
  --model AR_GRM --dataset AmazonReviews2014CleanGR \
  --checkpoint "$repo_dir/saved/AmazonReviews2014CleanGR_beauty14_full_opq_ar_finetune_control_v1/pytorch_model.bin" \
  --config "$repo_dir/experiments/canonical_full/beauty14_full.yaml" \
  --config "$repo_dir/experiments/grid_controls/beauty14_free_form.yaml" \
  --split test --no-diagnostics \
  --output "$run_dir/beauty14_free_form_test.json" \
  > "$run_dir/beauty14_free_form_test.log" 2>&1 &
echo $! > "$run_dir/beauty14_free_form_test.pid"

nohup env CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  "$python_bin" "$repo_dir/main.py" \
  --model AR_GRM --dataset AmazonReviews2014CleanGR \
  --config "$repo_dir/experiments/canonical_full/beauty14_full.yaml" \
  --config "$repo_dir/experiments/grid_controls/beauty14_level_heads_only.yaml" \
  > "$run_dir/beauty14_level_heads_only_v1.log" 2>&1 &
echo $! > "$run_dir/beauty14_level_heads_only_v1.pid"

echo "Launched free-form evaluation and level-head adaptation."
