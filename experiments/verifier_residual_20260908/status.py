#!/usr/bin/env python3
"""Read-only tmux dashboard for residual-verifier runs."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
path = root / 'suite_status.json'
if not path.exists():
    print('Waiting for residual supervisor:', root)
    raise SystemExit(0)
state = json.loads(path.read_text())
print('Residual verifier:', state['state'], state.get('phase', ''))
print('Updated UTC:', state['updated_utc'])
if state.get('error'):
    print('ERROR:', state['error'])
for job in state.get('jobs', []):
    progress = Path(job['output']) / 'progress.json'
    detail = json.loads(progress.read_text()) if progress.exists() else {}
    if 'examples' in detail:
        label = f"{detail.get('state')}: {detail['examples']:,}/{detail['total']:,}"
    elif 'batch' in detail:
        label = f"epoch {detail['epoch']} batch {detail['batch']}/{detail['total_batches']} loss {detail['loss']:.4f}"
    else:
        label = detail.get('state', 'initializing')
    print(f"GPU {job['gpu']} | {job['label']} | {label}")
for arm in ('causal', 'bidirectional'):
    history = root / 'full' / arm / 'history.json'
    if history.exists():
        row = json.loads(history.read_text())[-1]
        m = row['validation']['selected']
        print(f"{arm}: epoch {row['epoch']} N@10={m['ndcg@10']:.6f} R@10={m['recall@10']:.6f} "
              f"a={row['validation']['selected_alpha']} residual_rms={row['validation']['residual_rms']:.3f}")
