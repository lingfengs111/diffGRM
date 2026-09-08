#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/lingfengs111/codes/GR_variant/DiffGRM"
python_bin="/home/lingfengs111/miniconda3/envs/diffgrm/bin/python"
output_dir="$repo_dir/runs/motivation_2x2"

training_pids=(2393963 2393965 2396500 2396501)
training_markers=(
    beauty14_rq_cf_diff_v1
    beauty14_opq_cf_diff_v1
    beauty14_rq_cf_ar_constrained_v2
    beauty14_opq_cf_ar_constrained_v2
)

process_matches() {
    local pid="$1"
    local marker="$2"
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    tr '\0' ' ' <"/proc/$pid/cmdline" | rg -q --fixed-strings "$marker"
}

while true; do
    active=0
    for idx in "${!training_pids[@]}"; do
        if process_matches "${training_pids[$idx]}" "${training_markers[$idx]}"; then
            active=1
        fi
    done
    [[ "$active" -eq 0 ]] && break
    sleep 30
done

run_verifier() {
    local gpu="$1"
    local sid_name="$2"
    local sid_config="$3"
    local ar_run="$4"
    local diff_run="$5"
    local result_path="$output_dir/verifier_${sid_name}_multiorder.json"
    local log_path="$output_dir/verifier_${sid_name}_multiorder.log"

    [[ -f "$repo_dir/saved/AmazonReviews2014_${ar_run}/pytorch_model.bin" ]]
    [[ -f "$repo_dir/saved/AmazonReviews2014_${diff_run}/pytorch_model.bin" ]]

    env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
        "$python_bin" "$repo_dir/scripts/evaluate_proposal_verifier.py" \
        --dataset=AmazonReviews2014 \
        --common-config="$repo_dir/experiments/motivation_2x2/common.yaml" \
        --sid-config="$repo_dir/experiments/motivation_2x2/$sid_config" \
        --ar-config="$repo_dir/experiments/motivation_2x2/ar.yaml" \
        --diffusion-config="$repo_dir/experiments/motivation_2x2/diffusion.yaml" \
        --ar-checkpoint="$repo_dir/saved/AmazonReviews2014_${ar_run}/pytorch_model.bin" \
        --diffusion-checkpoint="$repo_dir/saved/AmazonReviews2014_${diff_run}/pytorch_model.bin" \
        --proposal-k=32 \
        --output-k=10 \
        --batch-size=8 \
        '--decode-orders=0,1,2,3;2,3,0,1;3,2,1,0;1,0,3,2' \
        --output="$result_path" \
        >"$log_path" 2>&1
}

cd "$repo_dir"
run_verifier 0 rq rq_cf.yaml \
    beauty14_rq_cf_ar_constrained_v2 beauty14_rq_cf_diff_v1 &
rq_pid=$!
run_verifier 2 opq opq_cf.yaml \
    beauty14_opq_cf_ar_constrained_v2 beauty14_opq_cf_diff_v1 &
opq_pid=$!
echo "verifiers launched: rq_pid=$rq_pid opq_pid=$opq_pid"
wait "$rq_pid" "$opq_pid"
