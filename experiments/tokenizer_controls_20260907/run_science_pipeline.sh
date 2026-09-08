#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU rqk4_esm|rqk4_hungarian|rqk3_hungarian|opq4_hungarian|rq2opq2_hungarian}"
arm="${2:?usage: $0 GPU ARM}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python

case "$arm" in
  rqk4_esm)
    sid_config=experiments/tokenizer_controls_20260907/science23_rqkmeans4_esm.yaml
    ;;
  rqk4_hungarian)
    sid_config=experiments/tokenizer_controls_20260907/science23_rqkmeans4_hungarian.yaml
    ;;
  rqk3_hungarian)
    sid_config=experiments/tokenizer_controls_20260907/science23_rqkmeans3_hungarian.yaml
    ;;
  opq4_hungarian)
    sid_config=experiments/tokenizer_controls_20260907/science23_opq4_hungarian.yaml
    ;;
  rq2opq2_hungarian)
    sid_config=experiments/tokenizer_controls_20260907/science23_rq2opq2_hungarian.yaml
    ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

root="$repo/runs/tokenizer_controls_20260907/$arm"
ar_run="science23_${arm}_ar_l20_v1"
ar_ckpt="$repo/saved/AmazonReviews2023CleanGR_${ar_run}/pytorch_model.bin"
result="$root/onepass_pairwise_ar/result.json"
mkdir -p "$root"
cd "$repo"

if [[ -s "$result" ]]; then
  echo "already complete: $result"
  exit 0
fi
pipeline_pid="$root/.pipeline.pid"
if [[ -s "$pipeline_pid" ]]; then
  active_pid="$(tr -d '[:space:]' <"$pipeline_pid")"
  if [[ "$active_pid" =~ ^[0-9]+$ ]] && kill -0 "$active_pid" 2>/dev/null; then
    echo "already running: $arm pid=$active_pid"
    exit 0
  fi
  rm -f "$pipeline_pid"
fi
printf '%s\n' "$$" >"$pipeline_pid"
trap 'rm -f "$pipeline_pid"' EXIT
if [[ ! -e "$repo/runs/tokenizer_controls_20260907/SIDS_READY" ]]; then
  echo "SID preparation marker is missing" >&2
  exit 3
fi

if [[ ! -s "$ar_ckpt" ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" main.py \
    --model=AR_GRM --dataset=AmazonReviews2023CleanGR \
    --config=experiments/amazon23_domains/common.yaml \
    --config="$sid_config" \
    --config=experiments/canonical_full/ar_constrained.yaml \
    --run_id="$ar_run" >"$root/ar.log" 2>&1
fi
test -s "$ar_ckpt"

CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false "$python_bin" \
  scripts/train_parallel_opq_drafter.py \
  --dataset AmazonReviews2023CleanGR \
  --common-config experiments/amazon23_domains/common.yaml \
  --sid-config "$sid_config" \
  --ar-config experiments/canonical_full/ar_constrained.yaml \
  --diffusion-config experiments/music23_transfer/guided_decoder.yaml \
  --ar-checkpoint "$ar_ckpt" \
  --backbone-architecture encoder_four_head \
  --backbone-initialization random --encoder-head-n-layer 4 \
  --variant pairwise --conditioner diffusion_encoder --pair-rank 51 \
  --epochs 180 --patience 20 --min-epochs 16 \
  --batch-size 256 --eval-batch-size 32 \
  --backbone-lr 0.0003 --selector-lr 0.001 \
  --weight-decay 0.0001 --token-loss-weight 0.1 \
  --proposal-k 72 --seed 2026 \
  --fusion-alphas 0,0.1,0.25,0.5,0.75,0.9,1 \
  --retain-candidate-checkpoint --dump-selected-ranks \
  --output-dir "$root/onepass_pairwise_ar" >"$root/onepass.log" 2>&1

test -s "$result"
echo "complete: $arm"
