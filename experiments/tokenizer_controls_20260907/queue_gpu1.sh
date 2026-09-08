#!/usr/bin/env bash
set -euo pipefail
here=/home/lingfengs111/codes/GR_variant/DiffGRM/experiments/tokenizer_controls_20260907
"$here/wait_for_slot.sh" 1
"$here/run_science_pipeline.sh" 1 rqk4_esm
"$here/run_science_pipeline.sh" 1 opq4_hungarian

