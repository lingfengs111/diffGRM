#!/usr/bin/env bash
set -euo pipefail
here=/home/lingfengs111/codes/GR_variant/DiffGRM/experiments/tokenizer_controls_20260907
"$here/run_psid_no_latent.sh" 3
"$here/run_science_pipeline.sh" 3 rqk4_hungarian

