#!/usr/bin/env python3
"""GPU-aware sequential gates with matched two-arm training in tmux."""
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
import yaml


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Suite:
    def __init__(self, root):
        self.root = root
        self.source = root / 'source'
        self.script = self.source / 'scripts/train_sid_rank_verifier.py'
        self.children = []

    def status(self, state, **fields):
        write(self.root / 'suite_status.json', dict(
            state=state, supervisor_pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(), **fields))

    def spawn(self, label, gpu, argv, output):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(self.script), *map(str, argv)]
        write(output / 'command.json', {'argv': command, 'gpu': gpu})
        with (output / 'run.log').open('a') as log:
            child = subprocess.Popen(command, cwd=self.source, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
        self.children.append(child)
        return {'label': label, 'gpu': gpu, 'process': child, 'output': str(output)}

    def wait(self, phase, jobs):
        while True:
            details = [{k: v for k, v in job.items() if k != 'process'} |
                       {'pid': job['process'].pid, 'exit_code': job['process'].poll()}
                       for job in jobs]
            self.status('running', phase=phase, jobs=details)
            failed = [j for j in details if j['exit_code'] not in (None, 0)]
            if failed:
                raise RuntimeError(f'job failed; downstream stages stopped: {failed}')
            if all(j['exit_code'] == 0 for j in details):
                return
            time.sleep(10)

    def prepare(self, domain, gpu, smoke=False):
        stage = 'smoke' if smoke else 'full'
        cache = self.root / stage / domain / 'cache'
        inputs = self.root / 'inputs' / domain
        runtime = self.root / 'runtime'
        runtime.mkdir(exist_ok=True)
        config = yaml.safe_load((inputs / 'domain.yaml').read_text())
        config['sid_override_path'] = str(inputs / 'sid.sem_ids')
        config['cache_dir'] = str(self.root / 'tokenizer_cache')
        config_path = runtime / f'{domain}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        args = ['prepare', '--cache', cache, '--common-config', inputs / 'common.yaml',
                '--sid-config', config_path, '--ar-config', inputs / 'ar.yaml',
                '--ar-checkpoint', inputs / 'ar.pt', '--drafter-checkpoint', inputs / 'drafter.pt',
                '--reference-result', inputs / 'reference.json', '--prepare-batch-size', '64',
                '--splits', 'val,train']
        if smoke:
            args += ['--limit', '128', '--prepare-batch-size', '32']
        return self.spawn(f'{stage}/{domain}/prepare', gpu, args, cache)

    def arms(self, domain, smoke=False):
        stage = 'smoke' if smoke else 'full'
        cache = self.root / stage / domain / 'cache'
        jobs = []
        for gpu, attention in enumerate(('causal', 'bidirectional')):
            output = self.root / stage / domain / attention
            args = ['train', '--cache', cache, '--output', output, '--attention', attention]
            if smoke:
                args += ['--epochs', '2']
            jobs.append(self.spawn(f'{stage}/{domain}/{attention}', gpu, args, output))
        return jobs

    def compare(self, domain):
        base = self.root / 'full' / domain
        baseline = json.loads((base / 'cache/baseline_val.json').read_text())
        results = {a: json.loads((base / a / 'result.json').read_text())['validation']
                   for a in ('causal', 'bidirectional')}
        settings = {a: json.loads((base / a / 'settings.json').read_text()) for a in results}
        if settings['causal']['initial_head_sha256'] != settings['bidirectional']['initial_head_sha256']:
            raise RuntimeError('B/C readout initialization differs')
        if settings['causal']['cache_metadata_sha256'] != settings['bidirectional']['cache_metadata_sha256']:
            raise RuntimeError('B/C candidate caches differ')
        qualified = [a for a, r in results.items()
                     if r['selected']['ndcg@10'] >= baseline['selected']['ndcg@10'] + .0002
                     and r['selected']['recall@10'] >= baseline['selected']['recall@10']]
        report = {'domain': domain, 'baseline': baseline, 'arms': results,
                  'qualified': qualified, 'rule': 'fused validation NDCG@10 >= A + .0002 and Recall@10 >= A',
                  'interpretation': 'single-seed validation screening; no independent test evidence'}
        ranks = {'baseline': np.load(base / 'cache/baseline_val_ranks.npz')}
        ranks.update({a: np.load(base / a / 'validation_ranks.npz') for a in results})
        report['migration'] = {}
        for arm, r in ranks.items():
            if not np.array_equal(r['target_rows'], ranks['baseline']['target_rows']):
                raise RuntimeError('rank dump sample order mismatch')
            d, f = r['drafter_rank'], r['fused_rank']
            report['migration'][arm] = {
                f'{lo}_{hi}': {'examples': int(((d >= lo) & (d <= hi)).sum()),
                              'top10_hits': int(((d >= lo) & (d <= hi) & (f <= 10)).sum())}
                for lo, hi in ((1, 10), (11, 32), (33, 72))}
        rng = np.random.default_rng(20260908)
        report['descriptive_paired_intervals'] = {}
        for left, right in (('causal', 'baseline'), ('bidirectional', 'baseline'),
                            ('bidirectional', 'causal')):
            row = {}
            for metric in ('ndcg', 'recall'):
                def contributions(values):
                    return (values <= 10).astype(float) if metric == 'recall' else np.where(
                        values <= 10, 1. / np.log2(values + 1), 0.)
                delta = contributions(ranks[left]['fused_rank']) - contributions(ranks[right]['fused_rank'])
                values, counts = np.unique(delta, return_counts=True)
                means = rng.multinomial(len(delta), counts / counts.sum(), size=2000) @ values / len(delta)
                row[f'{metric}@10'] = {'delta': float(delta.mean()),
                                      'ci95': np.quantile(means, [.025, .975]).tolist()}
            report['descriptive_paired_intervals'][f'{left}_minus_{right}'] = row
        write(self.root / f'{domain}_decision.json', report)
        return qualified

    def run(self):
        manifest = json.loads((self.root / 'source_manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha256(self.root / name) != digest:
                raise RuntimeError(f'snapshot hash changed: {name}')
        # Refuse to take GPUs with live compute jobs. GPU 3 belongs to another suite.
        query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                         '--format=csv,noheader'], text=True).splitlines()
        uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                         '--format=csv,noheader'], text=True).splitlines()
        if any(uuid in query for uuid in uuids[:3]):
            raise RuntimeError('GPUs 0/1/2 must be free before this suite starts')
        self.wait('smoke/preparation', [self.prepare('video23', 0, smoke=True)])
        self.wait('smoke/training', self.arms('video23', smoke=True))
        write(self.root / 'smoke_passed.json', {'two_epochs_both_arms': True,
              'warmup_and_decoder_adaptation': True, 'completed_utc': datetime.now(timezone.utc).isoformat()})
        self.wait('full/preparation', [self.prepare('video23', 0), self.prepare('science23', 2)])
        self.wait('full/video23', self.arms('video23'))
        qualified = self.compare('video23')
        if qualified:
            self.wait('full/science23', self.arms('science23'))
            self.compare('science23')
        self.status('complete', video_qualified=qualified, science_trained=bool(qualified),
                    next_step='Review validation findings before test/multi-seed/mining/retrieval controls.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'suite_status.json').exists():
        raise FileExistsError('suite already started; inspect its state and use a fresh root for retry')
    suite = Suite(root)
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
