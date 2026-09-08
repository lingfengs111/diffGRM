#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
diff_repo=/home/lingfengs111/codes/GR_variant/DiffGRM
latte_repo=/home/lingfengs111/codes/GR_variant/Latte
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_science/raw_core5
sid="$diff_repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed/sentence-t5-base_latte-meta_pca192_RQKMEANS3x256_psid.sem_ids"
output="$diff_repo/runs/tokenizer_controls_20260907/psid_rqk3_esm_no_latent"

mkdir -p "$output"
if [[ -s "$output/result.json" ]]; then
  echo "already complete: $output/result.json"
  exit 0
fi
cd "$latte_repo"
CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  run_psid_aligned.py \
  --category Industrial_and_Scientific \
  --data-dir "$data" \
  --sid-override-path "$sid" \
  --output-dir "$output" --epochs 180 --patience 20 --seed 2026 \
  >"$output/train.log" 2>&1
test -s "$output/result.json"

