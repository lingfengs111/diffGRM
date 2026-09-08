#!/usr/bin/env bash
set -euo pipefail
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
root="$repo/runs/history_interest_20260907/v1"
python_bin=/home/lingfengs111/miniconda3/envs/diffgrm/bin/python
mode="${1:?usage: launch.sh smoke|full}"
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
cd "$repo"
test -s "$root/source/experiments/history_interest_20260907/run_suite.py"
if [[ "$mode" == full ]]; then
  "$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["max_absolute_error"] <= 1e-6' "$root/parity_check.json"
fi
if [[ -f "$root/supervisor.pid" ]]; then
  prior_pid=$(cat "$root/supervisor.pid")
  if kill -0 "$prior_pid" 2>/dev/null; then
    echo "Supervisor still alive: $prior_pid" >&2
    exit 1
  fi
fi
nohup setsid "$python_bin" "$root/source/experiments/history_interest_20260907/run_suite.py" \
  --suite-root "$root" --mode "$mode" > "$root/${mode}_supervisor.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$root/supervisor.pid"
for attempt in {1..20}; do
  if [[ -s "$root/suite_status.json" ]] && rg -q "\"supervisor_pid\": $pid," "$root/suite_status.json"; then
    break
  fi
  sleep 0.25
done
kill -0 "$pid"
echo "Started $mode supervisor PID $pid; log: $root/${mode}_supervisor.log"
