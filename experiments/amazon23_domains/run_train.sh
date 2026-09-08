#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {video|music|science|office} {DIFF_GRM|AR_GRM} [gpu]" >&2
  exit 2
fi

domain="$1"
model="$2"
gpu="${3:-0}"
max_history_len="${MAX_HISTORY_LEN:-20}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/lingfengs111/miniconda3/envs/diffgrm/bin/python}"
common="$repo_dir/experiments/amazon23_domains/common.yaml"

case "$domain" in
  video)
    domain_config="$repo_dir/experiments/amazon23_domains/video23.yaml"
    run_prefix="video23"
    ;;
  music)
    domain_config="$repo_dir/experiments/amazon23_domains/music23.yaml"
    run_prefix="music23"
    ;;
  science)
    domain_config="$repo_dir/experiments/amazon23_domains/science23.yaml"
    run_prefix="science23"
    ;;
  office)
    domain_config="$repo_dir/experiments/amazon23_domains/office23.yaml"
    run_prefix="office23"
    ;;
  *)
    echo "Unknown domain: $domain" >&2
    exit 2
    ;;
esac

config_args=(--config="$common" --config="$domain_config")
case "$model" in
  DIFF_GRM)
    default_run_id="${run_prefix}_full_opq_cf_diff_guided_l${max_history_len}_v1"
    ;;
  AR_GRM)
    default_run_id="${run_prefix}_full_opq_cf_ar_l${max_history_len}_v1"
    config_args+=(--config="$repo_dir/experiments/canonical_full/ar_constrained.yaml")
    ;;
  *)
    echo "Unknown model: $model" >&2
    exit 2
    ;;
esac

run_id="${RUN_ID:-$default_run_id}"
override_args=()
override_args+=(
  --max_history_len="$max_history_len"
  --max_hist_len="$max_history_len"
)
if [[ "$max_history_len" != "20" ]]; then
  echo "WARNING: Amazon standard max history is 20; running explicit L${max_history_len}." >&2
fi
if [[ -n "${ITEM_TEXTS_FILE:-}" ]]; then
  override_args+=(--item_texts_file="$ITEM_TEXTS_FILE")
fi
if [[ -n "${METADATA_CACHE_TAG:-}" ]]; then
  override_args+=(--metadata_cache_tag="$METADATA_CACHE_TAG")
fi

output_dir="$repo_dir/runs/amazon23_domains/$domain"
mkdir -p "$output_dir"
echo "Starting domain=$domain model=$model gpu=$gpu run_id=$run_id max_history_len=$max_history_len"
cd "$repo_dir"
env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
  "$python_bin" "$repo_dir/main.py" \
    --model="$model" \
    --dataset=AmazonReviews2023CleanGR \
    "${config_args[@]}" \
    "${override_args[@]}" \
    --run_id="$run_id" \
    2>&1 | tee "$output_dir/$run_id.log"
