#!/usr/bin/env python3
"""Small read-only dashboard for the suite's tmux status window."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
status_path = root / 'suite_status.json'
if not status_path.exists():
    print('Waiting for supervisor:', root)
    raise SystemExit(0)
status = json.loads(status_path.read_text())
print('Verifier architecture:', status['state'], status.get('phase', ''))
print('Updated UTC:', status['updated_utc'])
print('Run:', root)
if status.get('error'):
    print('ERROR:', status['error'])
for job in status.get('jobs', []):
    path = Path(job['output']) / 'progress.json'
    detail = json.loads(path.read_text()) if path.exists() else {}
    if 'examples' in detail:
        progress = f"{detail.get('stage')}: {detail['examples']:,}/{detail['total']:,}"
    elif 'batch' in detail:
        progress = f"epoch {detail['epoch']}, batch {detail['batch']}/{detail['total_batches']}, loss {detail['loss']:.4f}"
    else:
        progress = detail.get('state', 'initializing / tokenizing')
    print(f"GPU {job['gpu']} | {job['label']} | PID {job['pid']} | {progress}")
for domain in ('video23', 'science23'):
    for arm in ('causal', 'bidirectional'):
        path = root / 'full' / domain / arm / 'history.json'
        if path.exists():
            record = json.loads(path.read_text())[-1]
            report = record['validation']
            print(f"{domain}/{arm}: epoch {record['epoch']} validation N@10={report['selected']['ndcg@10']:.6f} "
                  f"R@10={report['selected']['recall@10']:.6f} alpha={report['selected_alpha']}")
print('GPU 3 is reserved for the pre-existing task.')
