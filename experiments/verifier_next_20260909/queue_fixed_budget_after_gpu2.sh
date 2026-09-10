#!/usr/bin/env bash
set -euo pipefail

repo=/home/lingfengs111/codes/GR_variant/DiffGRM
dependency="$repo/runs/verifier_next_20260909/proposal_aware_video23/last_decoder/COMPLETE"
while [[ ! -s "$dependency" ]]; do
  sleep 30
done
cd "$repo"
exec bash experiments/verifier_next_20260909/run_fixed_budget_generation_union.sh 2
