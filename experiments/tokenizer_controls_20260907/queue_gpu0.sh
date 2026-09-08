#!/usr/bin/env bash
set -euo pipefail
here=/home/lingfengs111/codes/GR_variant/DiffGRM/experiments/tokenizer_controls_20260907
"$here/wait_for_slot.sh" 0
"$here/run_psid_no_latent.sh" 0
"$here/run_science_pipeline.sh" 0 rqk3_hungarian

