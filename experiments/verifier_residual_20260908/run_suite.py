#!/usr/bin/env python3
"""Run residual smoke, score caching, and matched arms without test access."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Suite:
    def __init__(self, root, base):
        self.root, self.base = root, base
        self.source = root / 'source'
        self.script = self.source / 'scripts/train_residual_verifier.py'
        self.children = []

    def status(self, state, **details):
        write(self.root / 'suite_status.json', {
            'state': state, 'supervisor_pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **details,
        })

    def spawn(self, label, gpu, arguments, output):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(self.script), *map(str, arguments)]
        write(output / 'command.json', {'argv': command, 'gpu': gpu})
        with (output / 'run.log').open('a') as log:
            process = subprocess.Popen(command, cwd=self.source, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        job = {'label': label, 'gpu': gpu, 'output': str(output), 'process': process}
        self.children.append(process)
        return job

    def wait(self, phase, jobs):
        while True:
            status = [{k: v for k, v in job.items() if k != 'process'} |
                      {'pid': job['process'].pid, 'exit_code': job['process'].poll()}
                      for job in jobs]
            self.status('running', phase=phase, jobs=status)
            failed = [j for j in status if j['exit_code'] not in (None, 0)]
            if failed:
                raise RuntimeError(f'job failed; subsequent stages stopped: {failed}')
            if all(j['exit_code'] == 0 for j in status):
                return
            time.sleep(10)

    def score_job(self, cache, gpu, limit=0):
        arguments = ['prepare-train-scores', '--base-cache', self.base,
                     '--score-cache', cache, '--eval-batch-size', '64', '--chunk-size', '16']
        if limit:
            arguments += ['--limit', limit]
        return self.spawn('prepare frozen train AR scores', gpu, arguments, cache)

    def arms(self, score_cache, prefix, epochs, limit=0):
        jobs = []
        for gpu, attention in enumerate(('causal', 'bidirectional')):
            output = self.root / prefix / attention
            arguments = ['train', '--base-cache', self.base, '--score-cache', score_cache,
                         '--output', output, '--attention', attention, '--epochs', epochs,
                         '--batch-size', '64', '--accumulate', '2', '--eval-batch-size', '64',
                         '--chunk-size', '16', '--head-lr', '0.0001', '--weight-decay', '0.0001',
                         '--residual-l2', '0.01', '--residual-cap', '2.0', '--seed', '2026']
            if limit:
                arguments += ['--limit', limit]
            jobs.append(self.spawn(f'{prefix}/{attention}', gpu, arguments, output))
        return jobs

    def decision(self):
        baseline = json.loads((self.base / 'baseline_val.json').read_text())
        result = {arm: json.loads((self.root / 'full' / arm / 'result.json').read_text())
                  for arm in ('causal', 'bidirectional')}
        settings = {arm: value['settings'] for arm, value in result.items()}
        if settings['causal']['initial_head_sha256'] != settings['bidirectional']['initial_head_sha256']:
            raise RuntimeError('B/C residual head initialization differs')
        report = {'baseline': baseline, 'arms': {arm: value['validation'] for arm, value in result.items()},
                  'best_epochs': {arm: value['best_epoch'] for arm, value in result.items()},
                  'rule': 'epoch 0 is eligible; B must exceed A, C must exceed B; one-seed validation only'}
        ranks = {'baseline': np.load(self.base / 'baseline_val_ranks.npz')}
        ranks.update({arm: np.load(self.root / 'full' / arm / 'validation_ranks.npz') for arm in result})
        report['migration'] = {}
        for arm, rank in ranks.items():
            if not np.array_equal(rank['target_rows'], ranks['baseline']['target_rows']):
                raise RuntimeError('validation examples differ')
            draft, fused = rank['drafter_rank'], rank['fused_rank']
            report['migration'][arm] = {
                'rescue': int(((draft > 10) & (draft <= 72) & (fused <= 10)).sum()),
                'harm': int(((draft <= 10) & (fused > 10)).sum()),
                'top10_hits': int((fused <= 10).sum()),
            }
        write(self.root / 'decision.json', report)

    def run(self):
        manifest = json.loads((self.root / 'source_manifest.json').read_text())
        for name, expected in manifest['source_files'].items():
            if digest(self.root / name) != expected:
                raise RuntimeError(f'source snapshot changed: {name}')
        if digest(self.base / 'ready.json') != manifest['base_ready_sha256']:
            raise RuntimeError('base cache ready manifest changed')
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                             '--format=csv,noheader'], text=True).splitlines()
        uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                         '--format=csv,noheader'], text=True).splitlines()
        if any(uuid in occupied for uuid in uuids[:3]):
            raise RuntimeError('GPUs 0/1/2 must be free before this suite starts')
        smoke_scores = self.root / 'smoke' / 'scores'
        self.wait('smoke/cache', [self.score_job(smoke_scores, 2, limit=256)])
        self.wait('smoke/train', self.arms(smoke_scores, 'smoke', epochs=2, limit=256))
        write(self.root / 'smoke_passed.json', {'zero_identity_and_two_epochs_both_arms': True,
                                                 'completed_utc': datetime.now(timezone.utc).isoformat()})
        full_scores = self.root / 'full' / 'scores'
        self.wait('full/cache', [self.score_job(full_scores, 2)])
        self.wait('full/train', self.arms(full_scores, 'full', epochs=8))
        self.decision()
        self.status('complete', next_step='Review validation result before seeds, Science, or test.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'suite_status.json').exists():
        raise FileExistsError('suite already started; inspect it or make a fresh root')
    manifest = json.loads((root / 'source_manifest.json').read_text())
    suite = Suite(root, Path(manifest['base_cache']))
    try:
        suite.run()
    except Exception as error:
        for child in suite.children:
            if child.poll() is None:
                child.terminate()
        suite.status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
