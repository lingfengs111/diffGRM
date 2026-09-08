#!/usr/bin/env bash
set -euo pipefail
here=/home/lingfengs111/codes/GR_variant/DiffGRM/experiments/tokenizer_controls_20260907
"$here/wait_for_slot.sh" 2
"$here/run_science_pipeline.sh" 2 rq2opq2_hungarian
"$here/run_dual_view.sh" 2 opq4_to_rqk3
"$here/run_dual_view.sh" 2 rqk3_to_opq4
