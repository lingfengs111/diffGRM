#!/usr/bin/env python3
"""Run Science screening, then a validation-selected Video transfer.

This supervisor owns only processes it launches. A failed wave prevents the
dependent stage from starting. It never kills or takes over unrelated jobs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_status(root, **fields):
    path = root / 'suite_status.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def wave(root, stage, domain, arms):
    children = []
    runner = root / 'source/experiments/history_interest_20260907/run_experiment.py'
    for gpu, arm in enumerate(arms):
        command = [sys.executable, str(runner), '--suite-root', str(root),
                   '--gpu', str(gpu), '--arm', arm, '--stage', stage, '--domain', domain]
        children.append((arm, subprocess.Popen(command)))
    write_status(root, phase=f'{stage}/{domain}', processes=[
        {'arm': arm, 'pid': child.pid, 'gpu': gpu} for gpu, (arm, child) in enumerate(children)])
    while any(child.poll() is None for _, child in children):
        time.sleep(15)
    failures = {arm: child.returncode for arm, child in children if child.returncode}
    if failures:
        raise RuntimeError(f'Wave failed; dependent stages halted: {failures}')


def metric(root, domain, arm, name):
    result = json.loads((root / 'full' / domain / arm / 'result.json').read_text())
    alpha = result['selected_fusion_alpha']
    return result['validation_with_verifier'][f'fused_a{alpha:g}_{name}'.replace('.', 'p')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    args = parser.parse_args()
    root = args.suite_root.resolve()
    write_status(root, supervisor_pid=os.getpid(), state='running', mode=args.mode)
    arms = ['pooled', 'mlp', 'attention', 'interest2']
    try:
        if args.mode == 'smoke':
            wave(root, 'smoke', 'science23', arms)
            wave(root, 'smoke', 'science23', ['interest4'])
            wave(root, 'parity', 'science23', ['pooled'])
            actual = json.loads((root / 'parity/science23/pooled/result.json').read_text())
            reference = json.loads(Path('/home/lingfengs111/codes/GR_variant/DiffGRM/runs/latte_comparison_pure/science23_opq4/onepass_pairwise_ar/result.json').read_text())
            differences = {k: abs(v - reference['validation_with_verifier'][k])
                           for k, v in actual['validation_with_verifier'].items()
                           if ('ndcg@' in k or 'recall@' in k) and k in reference['validation_with_verifier']}
            parity = {'max_absolute_error': max(differences.values()), 'metric_errors': differences}
            (root / 'parity_check.json').write_text(json.dumps(parity, indent=2))
            if parity['max_absolute_error'] > 1e-6:
                raise RuntimeError(f'Legacy checkpoint parity failed: {parity}')
        else:
            wave(root, 'full', 'science23', arms)
            baseline_n = metric(root, 'science23', 'pooled', 'ndcg@10')
            baseline_r = metric(root, 'science23', 'pooled', 'recall@10')
            # A practical screening threshold, not a significance claim.
            qualified = [a for a in ['attention', 'interest2']
                         if metric(root, 'science23', a, 'ndcg@10') >= baseline_n + .0001
                         and metric(root, 'science23', a, 'recall@10') >= baseline_r]
            winner = max(qualified, key=lambda a: metric(root, 'science23', a, 'ndcg@10')) if qualified else None
            video_arms = ['pooled'] + (['mlp', winner] if winner else [])
            decision = {'science_validation': {
                a: {m: metric(root, 'science23', a, m) for m in ['ndcg@10', 'recall@10']}
                for a in arms}, 'qualified': qualified, 'winner': winner,
                'video_arms': video_arms, 'rule': 'NDCG@10 >= pooled + 0.0001 AND Recall@10 >= pooled'}
            (root / 'transfer_decision.json').write_text(json.dumps(decision, indent=2))
            wave(root, 'full', 'video23', video_arms)
            # Architecture and all hyperparameters are fixed before test access.
            # Report baseline, capacity control and preselected new method.
            for domain in ['science23', 'video23']:
                wave(root, 'test', domain, video_arms)
                wave(root, 'candidate_test', domain, ['pooled'])
        write_status(root, state='complete', processes=[])
    except Exception as error:
        write_status(root, state='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
