#!/usr/bin/env python3
"""Train a zero-initialized, bounded correction on a frozen AR verifier.

All candidates retain their original AR path likelihood. The learned score is
only a residual. The epoch-zero checkpoint is therefore a byte-for-byte score
equivalent to the AR baseline and participates in validation selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from genrec.models.AR_GRM.candidate_rank_verifier import CandidateRankVerifier
from scripts.train_sid_rank_verifier import (
    ALPHAS, atomic_json, load_ar, load_arrays, score_report, sha256, stamp,
)


def load_limited_arrays(cache, split, limit):
    arrays = load_arrays(cache, split)
    if limit:
        return {name: values[:limit] for name, values in arrays.items()}
    return arrays


def validate_base_cache(cache):
    ready = json.loads((cache / 'ready.json').read_text())
    if sha256(cache / 'metadata.json') != ready['metadata_sha256']:
        raise RuntimeError('base candidate-cache metadata changed')
    for name, expected in ready['array_hashes'].items():
        if sha256(cache / name) != expected:
            raise RuntimeError(f'base candidate cache changed: {name}')
    return ready


def validate_score_cache(score_cache, base_cache, count, limit):
    ready_path = score_cache / 'ready.json'
    if not ready_path.exists():
        raise FileNotFoundError(f'missing prepared train AR scores: {ready_path}')
    ready = json.loads(ready_path.read_text())
    expected_base = sha256(base_cache / 'ready.json')
    if ready['base_ready_sha256'] != expected_base:
        raise RuntimeError('train AR scores use a different candidate cache')
    if ready['count'] != count or ready['limit'] != limit:
        raise RuntimeError('train AR score cache size differs from this run')
    score_path = score_cache / 'train_ar_scores.npy'
    if sha256(score_path) != ready['score_sha256']:
        raise RuntimeError('train AR score cache changed')
    return np.load(score_path, mmap_mode='r'), ready


@torch.no_grad()
def prepare_train_scores(args):
    base = args.base_cache.resolve()
    output = args.score_cache.resolve()
    validate_base_cache(base)
    if output.exists():
        # The tmux supervisor creates the directory only to place command.json
        # and its append-only run.log before the child starts. Treat precisely
        # that empty-cache state as new; any cache/progress artifact means an
        # interrupted or completed run and remains non-overwritable.
        allowed = {'command.json', 'run.log'}
        unexpected = [p.name for p in output.iterdir() if p.name not in allowed]
        if unexpected:
            raise FileExistsError(
                f'refusing to overwrite residual score cache {output}: {unexpected}'
            )
    else:
        output.mkdir(parents=True)
    arrays = load_limited_arrays(base, 'train', args.limit)
    count = len(arrays['targets'])
    catalog = torch.tensor(np.load(base / 'catalog.npy'), device='cuda', dtype=torch.long)
    ar, meta = load_ar(base, 'cuda')
    scores = np.lib.format.open_memmap(
        output / 'train_ar_scores.npy', mode='w+', dtype=np.float32,
        shape=arrays['candidate_rows'].shape,
    )
    for start in tqdm(range(0, count, args.eval_batch_size), desc='cache frozen AR train scores'):
        stop = min(start + args.eval_batch_size, count)
        history = torch.tensor(np.asarray(arrays['history'][start:stop]), device='cuda', dtype=torch.long)
        rows = torch.tensor(np.asarray(arrays['candidate_rows'][start:stop]), device='cuda', dtype=torch.long)
        target = torch.tensor(np.asarray(arrays['targets'][start:stop]), device='cuda', dtype=torch.long)
        if not torch.equal(rows[:, 0], target):
            raise RuntimeError('positive-first training cache invariant broken')
        current = ar.score_candidate_paths({'history_sid': history}, catalog[rows], chunk_size=args.chunk_size)
        if not torch.isfinite(current).all():
            raise FloatingPointError('non-finite frozen AR score')
        scores[start:stop] = current.cpu().numpy()
        if start and start % (args.eval_batch_size * 100) == 0:
            atomic_json(output / 'progress.json', {
                'state': 'caching_train_ar', 'examples': stop, 'total': count,
                'updated_utc': stamp(), 'pid': os.getpid(),
            })
    scores.flush()
    score_path = output / 'train_ar_scores.npy'
    metadata = {
        'created_utc': stamp(), 'base_cache': str(base),
        'base_ready_sha256': sha256(base / 'ready.json'),
        'ar_checkpoint': meta['ar_checkpoint'], 'ar_sha256': meta['ar_sha256'],
        'count': count, 'limit': args.limit, 'k': int(scores.shape[1]),
        'score_sha256': sha256(score_path),
    }
    atomic_json(output / 'ready.json', metadata)
    atomic_json(output / 'progress.json', {
        'state': 'complete', 'examples': count, 'total': count,
        'updated_utc': stamp(), 'pid': os.getpid(),
    })


def bounded_residual(raw, cap):
    # The AR likelihood has median within-query std about 1.25 on Video23.
    # A +/-2 log-likelihood correction can repair local order while not
    # permitting the new head to discard the base score wholesale.
    return float(cap) * torch.tanh(raw / float(cap))


@torch.no_grad()
def evaluate(ranker, arrays, base_scores, catalog, batch_size, chunk_size, cap):
    ranker.eval()
    residual = np.empty(arrays['candidate_rows'].shape, dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(residual), batch_size):
        stop = min(start + batch_size, len(residual))
        history = torch.tensor(np.asarray(arrays['history'][start:stop]), device='cuda', dtype=torch.long)
        rows = torch.tensor(np.asarray(arrays['candidate_rows'][start:stop]), device='cuda', dtype=torch.long)
        raw, _ = ranker(history, catalog[rows], chunk_size)
        residual[start:stop] = bounded_residual(raw, cap).cpu().numpy()
    if not np.isfinite(residual).all():
        raise FloatingPointError('non-finite residual score')
    total = np.asarray(base_scores, dtype=np.float32) + residual
    report, ranks = score_report(arrays['proposal_scores'], total,
                                 arrays['candidate_rows'], arrays['targets'])
    report.update(
        evaluation_seconds=time.perf_counter() - started,
        residual_rms=float(np.sqrt(np.mean(residual ** 2))),
        residual_abs_p95=float(np.quantile(np.abs(residual), .95)),
        residual_at_cap_fraction=float(np.mean(np.abs(residual) >= .9 * cap)),
    )
    return report, ranks, residual


def state_hash(ranker):
    digest = hashlib.sha256()
    for parameter in ranker.head.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, ranker, epoch, base_ready_sha256, ranks):
    temporary = path.with_suffix('.tmp.pt')
    torch.save({
        'model': ranker.state_dict(), 'epoch': epoch,
        'base_ready_sha256': base_ready_sha256, 'residual': True,
    }, temporary)
    temporary.replace(path)
    np.savez_compressed(path.with_name('validation_ranks.npz'), **ranks)


def train(args):
    base = args.base_cache.resolve()
    output = args.output.resolve()
    if output.exists():
        # See the analogous score-cache check: only the supervisor's command
        # and log placeholders are permitted before this child begins.
        allowed = {'command.json', 'run.log'}
        unexpected = [p.name for p in output.iterdir() if p.name not in allowed]
        if unexpected:
            raise FileExistsError(
                f'refusing to overwrite residual arm {output}: {unexpected}'
            )
    else:
        output.mkdir(parents=True)
    base_ready = validate_base_cache(base)
    train_arrays = load_limited_arrays(base, 'train', args.limit)
    val_arrays = load_limited_arrays(base, 'val', args.limit)
    train_base, score_ready = validate_score_cache(
        args.score_cache.resolve(), base, len(train_arrays['targets']), args.limit
    )
    val_base = np.load(base / 'val/ar_scores.npy', mmap_mode='r')
    if args.limit:
        val_base = val_base[:args.limit]
    catalog = torch.tensor(np.load(base / 'catalog.npy'), device='cuda', dtype=torch.long)
    torch.manual_seed(args.seed + 10)
    torch.cuda.manual_seed_all(args.seed + 10)
    ar, _ = load_ar(base, 'cuda')
    ranker = CandidateRankVerifier(ar, args.attention, zero_init_head=True).cuda()
    ranker.set_decoder_trainable(False)
    optimizer = torch.optim.AdamW(ranker.head.parameters(), lr=args.head_lr,
                                  weight_decay=args.weight_decay)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(
        n_train=len(train_arrays['targets']), n_val=len(val_arrays['targets']),
        parameters=sum(p.numel() for p in ranker.parameters()),
        head_parameters=sum(p.numel() for p in ranker.head.parameters()),
        frozen_ar_parameters=sum(p.numel() for p in ranker.ar.parameters()),
        initial_head_sha256=state_hash(ranker),
        base_ready_sha256=sha256(base / 'ready.json'),
        score_ready_sha256=sha256(args.score_cache / 'ready.json'),
    )
    atomic_json(output / 'settings.json', settings)
    baseline = json.loads((base / 'baseline_val.json').read_text())
    if args.limit:
        # A smoke run evaluates a prefix only, so construct its corresponding
        # frozen-AR baseline instead of comparing it to full validation.
        baseline, _ = score_report(val_arrays['proposal_scores'], val_base,
                                   val_arrays['candidate_rows'], val_arrays['targets'])
    initial, ranks, _ = evaluate(ranker, val_arrays, val_base, catalog,
                                 args.eval_batch_size, args.chunk_size, args.residual_cap)
    differences = {
        f'{alpha}/{metric}': abs(initial['by_alpha'][str(alpha)][metric] - baseline['by_alpha'][str(alpha)][metric])
        for alpha in ALPHAS for metric in ('ndcg@5', 'recall@5', 'ndcg@10', 'recall@10')
    }
    if max(differences.values()) > 1e-8:
        raise RuntimeError(f'zero residual does not reproduce AR baseline: {max(differences.values())}')
    atomic_json(output / 'initial_validation.json', initial | {'max_baseline_error': max(differences.values())})
    best = initial['selected']['ndcg@10']
    best_epoch = 0
    save_checkpoint(output / 'best.pt', ranker, best_epoch, settings['base_ready_sha256'], ranks)
    history = []
    for epoch in range(1, args.epochs + 1):
        ranker.train()
        random.seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        torch.cuda.manual_seed_all(args.seed + epoch)
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_arrays['targets']))
        total_batches = (len(order) + args.batch_size - 1) // args.batch_size
        total_loss, total_rank, total_penalty = [], [], []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, start in enumerate(tqdm(range(0, len(order), args.batch_size), desc=f'{args.attention} residual epoch {epoch}')):
            index = order[start:start + args.batch_size]
            history_sid = torch.tensor(np.asarray(train_arrays['history'][index]), device='cuda', dtype=torch.long)
            rows = torch.tensor(np.asarray(train_arrays['candidate_rows'][index]), device='cuda', dtype=torch.long)
            targets = torch.tensor(np.asarray(train_arrays['targets'][index]), device='cuda', dtype=torch.long)
            if not torch.equal(rows[:, 0], targets):
                raise RuntimeError('positive-first training cache invariant broken')
            raw, _ = ranker(history_sid, catalog[rows], args.chunk_size)
            residual = bounded_residual(raw, args.residual_cap)
            base_scores = torch.tensor(np.asarray(train_base[index]), device='cuda', dtype=residual.dtype)
            rank_loss = F.cross_entropy(base_scores + residual, torch.zeros(len(index), device='cuda', dtype=torch.long))
            penalty = residual.square().mean()
            loss = rank_loss + args.residual_l2 * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite residual training loss')
            group_start = (step // args.accumulate) * args.accumulate * args.batch_size
            group_examples = min(args.batch_size * args.accumulate, len(order) - group_start)
            (loss * len(index) / group_examples).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == total_batches:
                torch.nn.utils.clip_grad_norm_(ranker.head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss.append(float(loss.detach()))
            total_rank.append(float(rank_loss.detach()))
            total_penalty.append(float(penalty.detach()))
            if step % 100 == 0:
                atomic_json(output / 'progress.json', {
                    'state': 'training', 'epoch': epoch, 'batch': step + 1,
                    'total_batches': total_batches, 'loss': total_loss[-1],
                    'updated_utc': stamp(), 'pid': os.getpid(),
                })
        report, ranks, residual = evaluate(ranker, val_arrays, val_base, catalog,
                                           args.eval_batch_size, args.chunk_size, args.residual_cap)
        row = {
            'epoch': epoch, 'loss': float(np.mean(total_loss)),
            'rank_loss': float(np.mean(total_rank)), 'residual_l2': float(np.mean(total_penalty)),
            'validation': report, 'peak_cuda_memory_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'epoch_seconds': time.perf_counter() - started,
        }
        history.append(row)
        if report['selected']['ndcg@10'] > best:
            best, best_epoch = report['selected']['ndcg@10'], epoch
            save_checkpoint(output / 'best.pt', ranker, epoch, settings['base_ready_sha256'], ranks)
        atomic_json(output / 'history.json', history)
        atomic_json(output / 'progress.json', {
            'state': 'epoch_complete', 'epoch': epoch, 'best_epoch': best_epoch,
            'validation': report, 'updated_utc': stamp(), 'pid': os.getpid(),
        })
        print(json.dumps(row), flush=True)
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    if saved['base_ready_sha256'] != settings['base_ready_sha256']:
        raise RuntimeError('checkpoint base-cache mismatch')
    ranker.load_state_dict(saved['model'])
    selected, ranks, _ = evaluate(ranker, val_arrays, val_base, catalog,
                                  args.eval_batch_size, args.chunk_size, args.residual_cap)
    expected = initial if best_epoch == 0 else history[best_epoch - 1]['validation']
    if max(abs(selected['selected'][k] - expected['selected'][k]) for k in selected['selected']) > 1e-7:
        raise RuntimeError('selected residual checkpoint failed reload parity')
    np.savez_compressed(output / 'validation_ranks.npz', **ranks)
    atomic_json(output / 'result.json', {
        'state': 'complete', 'attention': args.attention, 'best_epoch': best_epoch,
        'validation': selected, 'baseline': baseline, 'settings': settings,
        'score_cache': score_ready, 'completed_utc': stamp(),
    })
    atomic_json(output / 'progress.json', {
        'state': 'complete', 'best_epoch': best_epoch,
        'updated_utc': stamp(), 'pid': os.getpid(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare-train-scores', 'train'))
    parser.add_argument('--base-cache', type=Path, required=True)
    parser.add_argument('--score-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--attention', choices=('causal', 'bidirectional'), default='causal')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--accumulate', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--residual-l2', type=float, default=.01)
    parser.add_argument('--residual-cap', type=float, default=2.)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU fallback')
    if min(args.epochs, args.batch_size, args.accumulate, args.eval_batch_size, args.chunk_size) < 1:
        raise ValueError('epochs and batch/chunk sizes must be positive')
    if args.limit < 0 or args.residual_l2 < 0 or args.residual_cap <= 0:
        raise ValueError('invalid residual settings')
    if args.mode == 'train' and args.output is None:
        raise ValueError('training needs --output')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mode == 'prepare-train-scores':
        prepare_train_scores(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
