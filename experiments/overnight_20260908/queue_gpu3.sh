#!/usr/bin/env bash
set -euo pipefail
here=/home/lingfengs111/codes/GR_variant/DiffGRM/experiments/overnight_20260908
"$here/run_science_capacity.sh" 3 compact_d128
"$here/run_science_capacity.sh" 3 depth_d256
"$here/run_science_capacity.sh" 3 width_d176
