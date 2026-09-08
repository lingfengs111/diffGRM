#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: $0 GPU}"
repo=/home/lingfengs111/codes/GR_variant/DiffGRM
suite="$repo/runs/history_interest_20260907/v1/suite_status.json"
sid_status="$repo/runs/tokenizer_controls_20260907/sid_prepare.status"
sid_ready="$repo/runs/tokenizer_controls_20260907/SIDS_READY"

echo "waiting for the active history-interest suite and SID preparation"
while true; do
  suite_state="$({ python - "$suite" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("missing")
else:
    try:
        print(json.loads(path.read_text()).get("state", "unknown"))
    except Exception:
        print("unknown")
PY
  } 2>/dev/null)"
  if [[ -e "$sid_ready" && "$suite_state" != running ]]; then
    break
  fi
  if [[ -s "$sid_status" ]] && grep -q '^failed' "$sid_status"; then
    echo "SID preparation failed: $(cat "$sid_status")" >&2
    exit 4
  fi
  printf '%s gpu=%s suite=%s sid_ready=%s\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$suite_state" "$([[ -e "$sid_ready" ]] && echo yes || echo no)"
  sleep 60
done

# The suite writes its terminal status before its last CUDA context always
# disappears. Require two consecutive low-memory observations.
free_count=0
while (( free_count < 2 )); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 300 )); then
    free_count=$((free_count + 1))
  else
    free_count=0
  fi
  printf '%s gpu=%s memory_used_mib=%s stable_free=%s/2\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$used" "$free_count"
  sleep 30
done
echo "gpu $gpu is ready"

