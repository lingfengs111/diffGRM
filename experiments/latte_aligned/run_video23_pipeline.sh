#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
data=/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_video_game/raw_core5
meta=/home/lingfengs111/codes/soft_patch_training/data/amazon23/video_game/meta_Video_Games.jsonl.gz
model=/home/lingfengs111/.cache/huggingface/hub/models--sentence-transformers--sentence-t5-base/snapshots/fc5d4628481afbbaaacd7af6bb07cf9d3865f781
processed="$repo/cache/AmazonReviews2023CleanGR/Video_Games/processed"
embedding="$processed/sentence-t5-base_latte_meta_raw_d768.sent_emb"
texts="$processed/latte_metadata_sentences.jsonl"
sid="$processed/sentence-t5-base_latte-meta_pca192_RQKMEANS3x256_psid.sem_ids"
root="$repo/runs/latte_aligned/video23"
mkdir -p "$root"
cd "$repo"

if [[ ! -s "$embedding" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$python_bin" scripts/encode_latte_metadata.py \
      --item-vocab "$data/item_vocab.csv" --metadata "$meta" \
      --output-embeddings "$embedding" --output-texts "$texts" \
      --model "$model" --batch-size 512 --device cuda --force \
      >"$root/encode.log" 2>&1
fi
test -s "$embedding"

if [[ ! -s "$sid" ]]; then
  "$python_bin" scripts/generate_latte_rqkmeans_sids.py \
    --data-dir "$data" --splits-dir splits \
    --embedding-path "$embedding" --output "$sid" \
    --embedding-dim 768 --pca-dim 192 \
    --n-codebooks 3 --codebook-size 256 --faiss-threads 32 \
    >"$root/sid.log" 2>&1
fi
test -s "$sid"
test -s "$sid.diagnostics.json"
bash experiments/latte_aligned/run_video23_latte_aligned.sh "$gpu" all

