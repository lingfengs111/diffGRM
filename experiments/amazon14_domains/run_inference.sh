#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 {beauty|cds|sports|toys} {DIFF_GRM|AR_GRM} CHECKPOINT [val|test] [gpu]" >&2
  exit 2
fi

domain="$1"
model="$2"
checkpoint="$3"
split="${4:-test}"
gpu="${5:-0}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/lingfengs111/miniconda3/envs/diffgrm/bin/python}"

if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 1
fi

case "$domain" in
  beauty) domain_config="$repo_dir/experiments/amazon14_domains/beauty14.yaml" ;;
  cds) domain_config="$repo_dir/experiments/amazon14_domains/cds14.yaml" ;;
  sports) domain_config="$repo_dir/experiments/amazon14_domains/sports14.yaml" ;;
  toys) domain_config="$repo_dir/experiments/amazon14_domains/toys14.yaml" ;;
  *) echo "Unknown domain: $domain" >&2; exit 2 ;;
esac

config_args=(
  --config "$repo_dir/experiments/amazon14_domains/common.yaml"
  --config "$domain_config"
)
if [[ "$model" == "DIFF_GRM" ]]; then
  config_args+=(--config "$repo_dir/experiments/canonical_full/diffusion_sequential.yaml")
elif [[ "$model" == "AR_GRM" ]]; then
  config_args+=(--config "$repo_dir/experiments/canonical_full/ar_constrained.yaml")
else
  echo "Unknown model: $model" >&2
  exit 2
fi

text_override_args=()
max_history_len="${MAX_HISTORY_LEN:-20}"
text_override_args+=(--max-history-len "$max_history_len")
if [[ "$max_history_len" != "20" ]]; then
  echo "WARNING: Amazon standard max history is 20; evaluating explicit L${max_history_len}." >&2
fi
if [[ -n "${ITEM_TEXTS_FILE:-}" ]]; then
  text_override_args+=(--item-texts-file "$ITEM_TEXTS_FILE")
fi
if [[ -n "${METADATA_CACHE_TAG:-}" ]]; then
  text_override_args+=(--metadata-cache-tag "$METADATA_CACHE_TAG")
fi

output_dir="$repo_dir/runs/amazon14_domains/$domain/inference"
mkdir -p "$output_dir"
checkpoint_name="$(basename "$(dirname "$checkpoint")")"
diagnostic_args=(--no-diagnostics)
if [[ "${DIAGNOSTICS:-0}" == "1" ]]; then
  diagnostic_args=()
fi

cd "$repo_dir"
env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
  "$python_bin" "$repo_dir/scripts/evaluate_checkpoint.py" \
    --model "$model" \
    --dataset AmazonReviews2014CleanGR \
    --checkpoint "$checkpoint" \
    "${config_args[@]}" \
    "${text_override_args[@]}" \
    --split "$split" \
    "${diagnostic_args[@]}" \
    --output "$output_dir/${checkpoint_name}_${split}.json"
