#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/canonical_full/multi_order"
gpu_id=2

ar_checkpoint="$repo_dir/saved/AmazonReviews2014CleanGR_beauty14_full_opq_cf_ar_v1/pytorch_model.bin"
diffusion_checkpoint="$repo_dir/saved/AmazonReviews2014CleanGR_beauty14_full_opq_cf_diff_v1/pytorch_model.bin"
snapshot_checkpoint="$output_dir/beauty14_diff_best_at_launch.bin"
diffusion_pid_file="$repo_dir/runs/canonical_full/beauty14_full_opq_cf_diff_v1.pid"

mkdir -p "$output_dir"
echo $$ > "$output_dir/launcher.pid"
cp --reflink=auto "$diffusion_checkpoint" "$snapshot_checkpoint"

run_eval() {
    local tag="$1"
    local checkpoint="$2"
    local proposal_k="$3"
    local decode_orders="$4"
    local result_path="$output_dir/${tag}.json"
    local log_path="$output_dir/${tag}.log"

    env CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/evaluate_proposal_verifier.py" \
        --dataset=AmazonReviews2014CleanGR \
        --common-config="$repo_dir/experiments/canonical_full/beauty14_full.yaml" \
        --ar-config="$repo_dir/experiments/canonical_full/ar_constrained.yaml" \
        --diffusion-config="$repo_dir/experiments/canonical_full/diffusion_sequential.yaml" \
        --ar-checkpoint="$ar_checkpoint" \
        --diffusion-checkpoint="$checkpoint" \
        --proposal-k="$proposal_k" \
        --metric-ks=5,10 \
        --output-k=10 \
        --batch-size=8 \
        --decode-orders="$decode_orders" \
        --output="$result_path" \
        >"$log_path" 2>&1
}

multi_orders='0,1,2,3;2,3,0,1;3,2,1,0;1,0,3,2'
single_order='0,1,2,3'

# Freeze the best diffusion checkpoint available at launch (epoch 52) so the
# comparison cannot silently change while the parent training process runs.
run_eval beauty14_full_multi_order_k32_launchbest "$snapshot_checkpoint" 32 "$multi_orders"
run_eval beauty14_full_single_order_k128_launchbest "$snapshot_checkpoint" 128 "$single_order"

# If training selects a newer checkpoint, repeat both equal-budget controls.
if [[ -f "$diffusion_pid_file" ]]; then
    diffusion_pid=$(<"$diffusion_pid_file")
    while kill -0 "$diffusion_pid" 2>/dev/null; do
        sleep 30
    done
fi

snapshot_sha=$(sha256sum "$snapshot_checkpoint" | awk '{print $1}')
final_sha=$(sha256sum "$diffusion_checkpoint" | awk '{print $1}')
if [[ "$snapshot_sha" != "$final_sha" ]]; then
    run_eval beauty14_full_multi_order_k32_final "$diffusion_checkpoint" 32 "$multi_orders"
    run_eval beauty14_full_single_order_k128_final "$diffusion_checkpoint" 128 "$single_order"
fi
