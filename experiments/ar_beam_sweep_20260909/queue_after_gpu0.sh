#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
wait_for="$repo/runs/verifier_next_20260909/capacity_reallocation/deep_d256/COMPLETE"
while [[ ! -s "$wait_for" ]]; do
  sleep 30
done
exec "$repo/experiments/ar_beam_sweep_20260909/run_video23.sh" 0
