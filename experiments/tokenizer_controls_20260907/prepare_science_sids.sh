#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_science/raw_core5
processed="$repo/cache/AmazonReviews2023CleanGR/Industrial_and_Scientific/processed"
embedding="$processed/sentence-t5-base_latte_meta_raw_d768.sent_emb"
run_root="$repo/runs/tokenizer_controls_20260907"
audit_root="$run_root/sid_audit"
status="$run_root/sid_prepare.status"

mkdir -p "$audit_root"
cd "$repo"
printf 'running\n' >"$status"
trap 'code=$?; if (( code != 0 )); then printf "failed exit=%s\n" "$code" >"$status"; fi' EXIT

generate() {
  local quantizer="$1"
  local digits="$2"
  local repair="$3"
  local output="$4"
  local raw="$5"
  local log="$6"
  if [[ -s "$output" && -s "$output.diagnostics.json" && -s "$raw" ]]; then
    echo "reuse $output"
    return
  fi
  "$python_bin" scripts/generate_latte_rqkmeans_sids.py \
    --data-dir "$data" --splits-dir splits_l20 \
    --embedding-path "$embedding" --embedding-dim 768 --pca-dim 192 \
    --quantizer "$quantizer" --n-codebooks "$digits" --codebook-size 256 \
    --repair-strategy "$repair" --faiss-threads 12 \
    --raw-codes-output "$raw" --output "$output" >"$log" 2>&1
}

compare_json() {
  "$python_bin" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

a, b = map(Path, sys.argv[1:])
left = json.loads(a.read_text())
right = json.loads(b.read_text())
if left != right:
    mismatches = sum(left.get(key) != right.get(key) for key in set(left) | set(right))
    raise SystemExit(f"SID regeneration mismatch: {a} vs {b}; item mismatches={mismatches}")
print(f"exact semantic mapping match: {a.name} == {b.name}")
PY
}

compare_raw() {
  "$python_bin" - "$1" "$2" <<'PY'
import numpy as np
import sys

a, b = sys.argv[1:]
left, right = np.load(a), np.load(b)
if not np.array_equal(left, right):
    raise SystemExit(f"raw quantizer assignments differ: {a} vs {b}")
print(f"raw assignments match exactly: {a} == {b}")
PY
}

# Regenerate the existing ESM anchors. These equality checks guarantee that
# ESM-vs-Hungarian changes collision repair only, not PCA/KMeans randomness.
rqk3_esm="$audit_root/rqk3_esm_regenerated.sem_ids"
rqk3_esm_raw="$audit_root/rqk3_esm_raw.npy"
generate rqkmeans 3 esm "$rqk3_esm" "$rqk3_esm_raw" "$audit_root/rqk3_esm.log"
compare_json \
  "$processed/sentence-t5-base_latte-meta_pca192_RQKMEANS3x256_psid.sem_ids" \
  "$rqk3_esm"

rqk3_hung="$processed/sentence-t5-base_latte-meta_pca192_RQKMEANS3x256_hungarian.sem_ids"
rqk3_hung_raw="$audit_root/rqk3_hungarian_raw.npy"
generate rqkmeans 3 hungarian "$rqk3_hung" "$rqk3_hung_raw" "$audit_root/rqk3_hungarian.log"
compare_raw "$rqk3_esm_raw" "$rqk3_hung_raw"

opq4_esm="$audit_root/opq4_esm_regenerated.sem_ids"
opq4_esm_raw="$audit_root/opq4_esm_raw.npy"
generate opq 4 esm "$opq4_esm" "$opq4_esm_raw" "$audit_root/opq4_esm.log"
compare_json \
  "$processed/sentence-t5-base_latte-meta_pca192_OPQ4x256_psid.sem_ids" \
  "$opq4_esm"

opq4_hung="$processed/sentence-t5-base_latte-meta_pca192_OPQ4x256_hungarian.sem_ids"
opq4_hung_raw="$audit_root/opq4_hungarian_raw.npy"
generate opq 4 hungarian "$opq4_hung" "$opq4_hung_raw" "$audit_root/opq4_hungarian.log"
compare_raw "$opq4_esm_raw" "$opq4_hung_raw"

# New four-coordinate tokenizers.
generate rqkmeans 4 esm \
  "$processed/sentence-t5-base_latte-meta_pca192_RQKMEANS4x256_esm.sem_ids" \
  "$audit_root/rqk4_esm_raw.npy" "$audit_root/rqk4_esm.log"
generate rqkmeans 4 hungarian \
  "$processed/sentence-t5-base_latte-meta_pca192_RQKMEANS4x256_hungarian.sem_ids" \
  "$audit_root/rqk4_hungarian_raw.npy" "$audit_root/rqk4_hungarian.log"
compare_raw "$audit_root/rqk4_esm_raw.npy" "$audit_root/rqk4_hungarian_raw.npy"

generate rq2opq2 4 hungarian \
  "$processed/sentence-t5-base_latte-meta_pca192_RQ2OPQ2x256_hungarian.sem_ids" \
  "$audit_root/rq2opq2_hungarian_raw.npy" "$audit_root/rq2opq2_hungarian.log"

printf 'complete\n' >"$status"
touch "$run_root/SIDS_READY"
echo "Science23 tokenizer controls are ready."

