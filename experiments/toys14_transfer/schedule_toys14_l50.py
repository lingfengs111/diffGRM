#!/usr/bin/env python3
"""Run the full-data Toys14 L50 paper comparison on every available GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


HOME = Path('/home/lingfengs111')
REPO = HOME / 'codes/GR_variant/DiffGRM'
ROOT = REPO / 'runs/toys14_transfer_l50/queue'
DONE = ROOT / 'done'
STATE = ROOT / 'state.json'
PROCESSED = REPO / 'cache/AmazonReviews2014CleanGR/Toys_and_Games/processed'

JOBS = [
    {
        'name': 'toys14_l50_ar', 'priority': 10,
        'command': 'bash experiments/toys14_transfer/run_toys14_ar_l50.sh {gpu}',
        'requires': [], 'depends': [],
        'outputs': [REPO / 'saved/AmazonReviews2014CleanGR_toys14_full_opq_cf_ar_l50_long_v1/pytorch_model.bin'],
    },
    {
        'name': 'toys14_l50_sasrec', 'priority': 20,
        'command': 'bash /home/lingfengs111/codes/GR/CleanGR/scripts/run_amazon14_toys_sasrec_l50.sh {gpu}',
        'requires': [], 'depends': [],
        'outputs': [HOME / 'codes/GR/CleanGR/outputs/amazon14_toys/sasrec_long_l50/eval/test_metrics.json'],
    },
    {
        'name': 'toys14_l50_rpg', 'priority': 30,
        'command': 'bash /home/lingfengs111/codes/GR_variant/RPG_KDD2025/experiments/run_toys14_sentence_t5_l50.sh {gpu}',
        'requires': [], 'depends': [], 'outputs': [],
    },
    {
        'name': 'toys14_l50_diffgrm', 'priority': 40,
        'command': 'bash experiments/toys14_transfer/run_toys14_diffgrm_l50.sh {gpu}',
        'requires': [
            PROCESSED / 'sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto.sem_ids',
            PROCESSED / 'item_id2tokens_sentence-t5-base_pca256_OPQ4,IVF1,PQ4x8_cfhungarian-auto_4d.npy',
        ],
        'depends': [],
        'outputs': [REPO / 'saved/AmazonReviews2014CleanGR_toys14_full_opq_cf_diff_guided_l50_long_v1/pytorch_model.bin'],
    },
    {
        'name': 'toys14_l50_onepass_pairwise', 'priority': 50,
        'command': 'bash experiments/toys14_transfer/run_toys14_onepass_l50.sh {gpu}',
        'requires': [], 'depends': ['toys14_l50_ar'],
        'outputs': [REPO / 'runs/toys14_transfer_l50/random_encoder4_pairwise_ar/converged/result.json'],
    },
    {
        'name': 'toys14_l50_onepass_unary', 'priority': 60,
        'command': 'bash experiments/toys14_transfer/run_toys14_unary_l50.sh {gpu}',
        'requires': [], 'depends': ['toys14_l50_ar'],
        'outputs': [REPO / 'runs/toys14_transfer_l50/random_encoder4_unary_ar/converged/result.json'],
    },
    {
        'name': 'toys14_l50_exact_mips', 'priority': 70,
        'command': 'bash experiments/toys14_transfer/run_toys14_exact_mips_l50.sh {gpu}',
        'requires': [], 'depends': ['toys14_l50_ar'],
        'outputs': [REPO / 'runs/toys14_transfer_l50/exact_mips_id_ar/result.json'],
    },
]


def session_exists(name: str) -> bool:
    return subprocess.run(
        ['tmux', 'has-session', '-t', name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def files_exist(paths) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def gpu_state():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True,
    )
    return {
        int(line.split(',')[0]): tuple(int(value.strip()) for value in line.split(',')[1:])
        for line in result.stdout.splitlines()
    }


def save(state):
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    temporary.replace(STATE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=int, default=20)
    parser.add_argument('--stable-free-seconds', type=int, default=30)
    args = parser.parse_args()
    DONE.mkdir(parents=True, exist_ok=True)
    state = {'jobs': {}}
    save(state)
    free_since = {}

    while True:
        for job in JOBS:
            entry = state['jobs'].get(job['name'])
            if not entry or entry['status'] != 'running':
                continue
            if session_exists(entry['session']):
                continue
            marker = DONE / f"{job['name']}.done"
            ok = marker.is_file() and files_exist(job['outputs'])
            entry.update(status='done' if ok else 'failed', finished_at=time.time())
            print(f"{job['name']} -> {entry['status']}", flush=True)

        if all(state['jobs'].get(job['name'], {}).get('status') in {'done', 'failed'} for job in JOBS):
            save(state)
            print('Toys14 L50 queue finished', flush=True)
            return

        observations = gpu_state()
        reserved = {
            entry['gpu'] for entry in state['jobs'].values()
            if entry.get('status') == 'running' and session_exists(entry['session'])
        }
        now = time.time()
        available = []
        for gpu, (used, util) in observations.items():
            if gpu in reserved or used >= 300 or util > 10:
                free_since.pop(gpu, None)
                continue
            free_since.setdefault(gpu, now)
            if now - free_since[gpu] >= args.stable_free_seconds:
                available.append(gpu)

        done_names = {
            name for name, entry in state['jobs'].items() if entry.get('status') == 'done'
        }
        ready = []
        for job in JOBS:
            if job['name'] in state['jobs']:
                continue
            if not set(job['depends']).issubset(done_names) or not files_exist(job['requires']):
                continue
            ready.append(job)
        ready.sort(key=lambda job: (job['priority'], job['name']))

        for gpu, job in zip(sorted(available), ready):
            marker = DONE / f"{job['name']}.done"
            marker.unlink(missing_ok=True)
            session = f"toys_l50_{job['name']}"
            command = job['command'].format(gpu=gpu)
            wrapped = (
                f"cd {shlex.quote(str(REPO))} && {command} && "
                f"touch {shlex.quote(str(marker))}"
            )
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, wrapped], check=True)
            state['jobs'][job['name']] = {
                'status': 'running', 'gpu': gpu, 'session': session,
                'command': command, 'started_at': time.time(),
            }
            free_since.pop(gpu, None)
            print(f"launched {job['name']} on GPU {gpu}", flush=True)
        save(state)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
