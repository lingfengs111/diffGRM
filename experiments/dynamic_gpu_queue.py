#!/usr/bin/env python3
"""Dependency-aware local GPU experiment scheduler.

The scheduler never moves or interrupts a running process.  It observes GPUs
that remain below both the memory and utilization thresholds, reserves them
for tmux jobs it launches, and dispatches the highest-priority ready task.
State and success markers make the queue restart-safe and prevent duplicate
launches.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REPO = Path('/home/lingfengs111/codes/GR_variant/DiffGRM')
STATE_DIR = REPO / 'runs' / 'dynamic_gpu_queue'
STATE_FILE = STATE_DIR / 'state.json'
DONE_DIR = STATE_DIR / 'done'

MUSIC_PROCESSED = (
    REPO / 'cache/AmazonReviews2023CleanGR/Musical_Instruments/processed'
)

JOBS = [
    {
        'name': 'music23_ar',
        'priority': 10,
        'command': 'bash experiments/music23_transfer/run_music23_ar.sh',
        'requires_files': [],
        'depends_jobs': [],
        'outputs': [
            REPO / 'saved/AmazonReviews2023CleanGR_music23_full_opq_cf_ar_l20_long_v1/pytorch_model.bin',
        ],
    },
    {
        'name': 'music23_diffgrm',
        'priority': 20,
        'command': 'bash experiments/music23_transfer/run_music23_diffgrm.sh',
        'requires_files': [
            MUSIC_PROCESSED / 'sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto.sem_ids',
            MUSIC_PROCESSED / 'item_id2tokens_sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto_4d.npy',
        ],
        # Cache readiness, rather than AR completion, lets both long model
        # trainings overlap without concurrent cache writers.
        'depends_jobs': [],
        'outputs': [
            REPO / 'saved/AmazonReviews2023CleanGR_music23_full_opq_cf_diff_guided_l20_long_v1/pytorch_model.bin',
        ],
    },
    {
        'name': 'video23_candidate_budget',
        'priority': 30,
        'command': 'bash experiments/capacity_fairness/eval_video23_half_candidate_budget.sh',
        'requires_files': [
            REPO / 'runs/sampled_catalog/video23_half_pairwise/uniform_corrected_k1024/result.json',
        ],
        'depends_jobs': [],
        'outputs': [
            REPO / 'runs/capacity_fairness/video23_half_2x2_d176/candidate_budget/k128/result.json',
        ],
    },
    {
        'name': 'music23_onepass_random',
        'priority': 40,
        'command': (
            'bash experiments/music23_transfer/'
            'run_music23_onepass_fusion.sh random'
        ),
        'requires_files': [],
        # The encoder-four-head backbone is initialized from scratch.  It
        # needs the frozen AR verifier but never loads a DiffGRM checkpoint.
        'depends_jobs': ['music23_ar'],
        'outputs': [
            REPO / 'runs/music23_transfer/random_encoder4_pairwise_ar/result.json',
        ],
    },
    {
        'name': 'music23_onepass_pretrained',
        'priority': 50,
        'command': (
            'bash experiments/music23_transfer/'
            'run_music23_onepass_fusion.sh pretrained'
        ),
        'requires_files': [],
        'depends_jobs': ['music23_ar', 'music23_diffgrm'],
        'outputs': [
            REPO / 'runs/music23_transfer/'
            'diff_pretrained_masked_pairwise_ar/result.json',
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=int, default=20)
    parser.add_argument('--stable-free-seconds', type=int, default=20)
    parser.add_argument('--max-used-mib', type=int, default=300)
    parser.add_argument('--max-utilization', type=int, default=10)
    return parser.parse_args()


def log(message):
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {message}', flush=True)


def load_state():
    if not STATE_FILE.exists():
        return {'jobs': {}}
    with STATE_FILE.open() as handle:
        return json.load(handle)


def save_state(state):
    temporary = STATE_FILE.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    temporary.replace(STATE_FILE)


def session_exists(name):
    return subprocess.run(
        ['tmux', 'has-session', '-t', name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def gpu_observations():
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    observations = {}
    for line in result.stdout.splitlines():
        index, used, utilization = [int(value.strip()) for value in line.split(',')]
        observations[index] = {'used_mib': used, 'utilization': utilization}
    return observations


def files_exist(paths):
    return all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths)


def success_marker(job_name):
    return DONE_DIR / f'{job_name}.done'


def reconcile(state):
    entries = state['jobs']
    for job in JOBS:
        name = job['name']
        entry = entries.get(name)
        if entry and entry.get('status') == 'running':
            if session_exists(entry['session']):
                continue
            if success_marker(name).exists() and files_exist(job['outputs']):
                entry.update(status='done', finished_at=time.time())
                log(f'DONE job={name} gpu={entry.get("gpu")}')
            else:
                entry.update(status='failed', finished_at=time.time())
                log(f'FAILED job={name}; tmux ended without success marker/output')
        elif not entry and success_marker(name).exists() and files_exist(job['outputs']):
            entries[name] = {'status': 'done', 'recovered': True}


def job_ready(job, state):
    entry = state['jobs'].get(job['name'])
    if entry and entry.get('status') in {'running', 'done', 'failed'}:
        return False
    if not files_exist(job['requires_files']):
        return False
    return all(
        state['jobs'].get(dependency, {}).get('status') == 'done'
        for dependency in job['depends_jobs']
    )


def launch(job, gpu, state):
    name = job['name']
    session = f'dq_{name}_0901'
    marker = success_marker(name)
    marker.unlink(missing_ok=True)
    command = (
        f'cd {shlex.quote(str(REPO))} && '
        f'export CUDA_VISIBLE_DEVICES={int(gpu)} && '
        f'{job["command"]} && '
        f'mkdir -p {shlex.quote(str(DONE_DIR))} && '
        f'touch {shlex.quote(str(marker))}'
    )
    subprocess.run(
        ['tmux', 'new-session', '-d', '-s', session, command],
        check=True,
    )
    state['jobs'][name] = {
        'status': 'running',
        'gpu': int(gpu),
        'session': session,
        'started_at': time.time(),
        'command': job['command'],
    }
    log(f'LAUNCH job={name} gpu={gpu} session={session}')


def main():
    args = parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = (STATE_DIR / 'scheduler.lock').open('w')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another dynamic GPU scheduler is already running')

    state = load_state()
    free_since = {}
    while True:
        reconcile(state)
        save_state(state)
        terminal = {
            state['jobs'].get(job['name'], {}).get('status') for job in JOBS
        }
        if all(status in {'done', 'failed'} for status in terminal):
            log('all queued jobs reached a terminal state')
            return

        observations = gpu_observations()
        reserved = {
            int(entry['gpu'])
            for entry in state['jobs'].values()
            if entry.get('status') == 'running'
            and session_exists(entry.get('session', ''))
        }
        now = time.time()
        available = []
        for gpu, observation in observations.items():
            externally_free = (
                observation['used_mib'] < args.max_used_mib
                and observation['utilization'] <= args.max_utilization
            )
            if gpu in reserved or not externally_free:
                free_since.pop(gpu, None)
                continue
            free_since.setdefault(gpu, now)
            if now - free_since[gpu] >= args.stable_free_seconds:
                available.append(gpu)

        ready = sorted(
            (job for job in JOBS if job_ready(job, state)),
            key=lambda job: (job['priority'], job['name']),
        )
        for gpu, job in zip(sorted(available), ready):
            launch(job, gpu, state)
            free_since.pop(gpu, None)
        save_state(state)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
