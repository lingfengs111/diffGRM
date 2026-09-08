#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/lingfengs111/miniconda3/envs/diffgrm/bin/python}"
run_id="video23_full_opq_cf_diff_guided_l20_v1"
checkpoint="$repo_dir/saved/AmazonReviews2023CleanGR_${run_id}/pytorch_model.bin"
baseline="$repo_dir/runs/canonical_full/video23_full_opq_cf_diff_guided_best_test.json"
checkpoint_name="$(basename "$(dirname "$checkpoint")")"
candidate="$repo_dir/runs/amazon23_domains/video/inference/${checkpoint_name}_test.json"
comparison="$repo_dir/runs/amazon23_domains/video/${run_id}_vs_l50.json"

if [[ ! -f "$baseline" ]]; then
  echo "L50 baseline not found: $baseline" >&2
  exit 1
fi

echo "Training Video23 guided DiffGRM with max_history_len=20 on GPU $gpu"
MAX_HISTORY_LEN=20 RUN_ID="$run_id" PYTHON_BIN="$python_bin" \
  bash "$repo_dir/experiments/amazon23_domains/run_train.sh" video DIFF_GRM "$gpu"

echo "Evaluating best L20 checkpoint: $checkpoint"
MAX_HISTORY_LEN=20 PYTHON_BIN="$python_bin" \
  bash "$repo_dir/experiments/amazon23_domains/run_inference.sh" \
    video DIFF_GRM "$checkpoint" test "$gpu"

echo "Comparing L20 test metrics with the canonical L50 baseline"
"$python_bin" "$repo_dir/scripts/compare_history_length_metrics.py" \
  --baseline "$baseline" \
  --candidate "$candidate" \
  --baseline-label L50 \
  --candidate-label L20 \
  --output "$comparison"
