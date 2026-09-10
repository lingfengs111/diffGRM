#!/usr/bin/env bash
set -euo pipefail
run_root="${1:?usage: launch.sh RUN_ROOT [TMUX_SESSION]}"
session="${2:-verifier_residual_0908}"
python_bin="${PYTHON_BIN:-/home/lingfengs111/miniconda3/envs/diffgrm/bin/python}"
run_root=$(realpath "$run_root")
runner="$run_root/source/experiments/verifier_residual_20260908/run_suite.py"
test -s "$runner"
test -x "$python_bin"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi
printf -v launch_command '%q %q --root %q >> %q 2>&1' \
  "$python_bin" "$runner" "$run_root" "$run_root/supervisor.log"
tmux new-session -d -s "$session" "$launch_command"
echo "Started tmux: $session"
echo "Status: $run_root/suite_status.json"
