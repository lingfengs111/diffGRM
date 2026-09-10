#!/usr/bin/env bash
set -euo pipefail
here=/home/lingfengs111/codes/GR_variant/DiffGRM/experiments/overnight_20260908
"$here/run_science_sampled.sh" 1 256
"$here/run_science_sampled.sh" 1 1024
"$here/run_science_sampled.sh" 1 4096
