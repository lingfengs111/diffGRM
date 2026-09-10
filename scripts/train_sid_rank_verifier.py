#!/usr/bin/env python3
"""Prepare immutable proposal caches and train matched whole-SID verifiers.

Screening only scores/optimizes train/validation examples. The inherited
dataset adapter also loads held-out rows for metadata bookkeeping.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.candidate_rank_verifier import CandidateRankVerifier

ALPHAS = (0., .1, .25, .5, .75, .9, 1.)


def drafter_runtime_settings(checkpoint):
    """Restore non-parameter settings; state_dict alone does not fix logits.

In the retained trainer, mtp_temperature scales even non-normalized logits.
Replacing it with the class default changes unary/pairwise balance and Top-K.
"""
    saved = checkpoint['args']
    required = {'backbone_architecture': 'encoder_four_head', 'history_head': 'pooled',
                'n_interests': 1, 'variant': 'pairwise', 'pair_rank': 51}
    if any(saved.get(k) != v for k, v in required.items()):
        raise ValueError('checkpoint is not the fixed pooled/pairwise-51 main-line drafter')
    temperature = float(saved['mtp_temperature'])
    if temperature <= 0:
        raise ValueError('invalid checkpoint logit temperature')
    return {'encoder_head_n_layer': int(saved['encoder_head_n_layer']),
            'encoder_head_normalize_logits': saved['training_objective'] == 'mtp_only',
            'encoder_head_logit_temperature': temperature}


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def normalize(scores):
    return (scores - scores.mean(1, keepdim=True)) / scores.std(
        1, keepdim=True, unbiased=False).clamp_min(1e-6)


def target_ranks(scores, rows, targets):
    order = scores.argsort(dim=1, descending=True, stable=True)
    hit = rows.gather(1, order).eq(targets[:, None])
    positions = torch.arange(1, rows.shape[1] + 1, device=rows.device)[None]
    return torch.where(hit, positions, rows.shape[1] + 1).min(1).values


def rank_metrics(ranks):
    return {f'{metric}@{k}': float(np.mean(
        (ranks <= k) if metric == 'recall' else
        np.where(ranks <= k, 1. / np.log2(ranks + 1), 0.)
    )) for k in (5, 10) for metric in ('ndcg', 'recall')}


def score_report(proposal, verification, rows, targets, batch_size=1024):
    ranks = {str(a): [] for a in ALPHAS}
    draft = []
    for start in range(0, len(targets), batch_size):
        stop = start + batch_size
        p = torch.tensor(np.asarray(proposal[start:stop]))
        v = torch.tensor(np.asarray(verification[start:stop]))
        r = torch.tensor(np.asarray(rows[start:stop]), dtype=torch.long)
        t = torch.tensor(np.asarray(targets[start:stop]), dtype=torch.long)
        draft.append(target_ranks(p, r, t).numpy())
        p, v = normalize(p), normalize(v)
        for a in ALPHAS:
            ranks[str(a)].append(target_ranks((1-a)*p + a*v, r, t).numpy())
    ranks = {a: np.concatenate(values) for a, values in ranks.items()}
    metrics = {a: rank_metrics(values) for a, values in ranks.items()}
    alpha = max(ALPHAS, key=lambda a: metrics[str(a)]['ndcg@10'])
    draft = np.concatenate(draft)
    fused = ranks[str(alpha)]
    covered = draft <= rows.shape[1]
    rescue = (draft > 10) & covered
    good = draft <= 10
    result = {
        'n_examples': len(targets), 'selected_alpha': alpha,
        'selected': metrics[str(alpha)], 'by_alpha': metrics,
        'verifier_only': metrics['1.0'], 'drafter_only': rank_metrics(draft),
        'candidate_recall': float(covered.mean()),
        'rescue_count': int(((fused <= 10) & rescue).sum()),
        'harm_count': int(((fused > 10) & good).sum()),
        'rescue_opportunities': int(rescue.sum()),
        'original_top10_hits': int(good.sum()),
    }
    return result, {'drafter_rank': draft, 'verifier_rank': ranks['1.0'],
                    'fused_rank': fused, 'target_rows': np.asarray(targets)}


def load_arrays(cache, split):
    directory = Path(cache) / split
    return {name: np.load(directory / f'{name}.npy', mmap_mode='r')
            for name in ('history', 'targets', 'candidate_rows', 'proposal_scores')}


def load_ar(cache, device):
    meta = json.loads((Path(cache) / 'metadata.json').read_text())
    config = dict(meta['ar_config'], constrained_beam=False)
    tokenizer = SimpleNamespace(**meta['tokenizer'])
    ar = AR_GRM(config, None, tokenizer).to(device)
    checkpoint = Path(meta['ar_checkpoint'])
    if sha256(checkpoint) != meta['ar_sha256']:
        raise RuntimeError('AR checkpoint changed after candidate preparation')
    ar.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return ar, meta


def prepare(args):
    from accelerate import Accelerator
    from genrec.diagnostics import catalog_codes
    from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
    from genrec.models.DIFF_GRM.encoder_head_drafter import EncoderOnlyFourHeadDrafter
    from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector, code_rows
    from genrec.utils import get_config, get_dataset
    from scripts.train_parallel_opq_drafter import one_pass_outputs

    root = args.cache
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'ready.json').exists():
        raise FileExistsError('refusing to regenerate a completed candidate cache')
    if (root / 'metadata.json').exists():
        raise FileExistsError('incomplete cache exists; use a fresh output directory')
    files = [str(args.common_config), str(args.sid_config)]
    ar_config = get_config('AR_GRM', 'AmazonReviews2023CleanGR',
                          files + [str(args.ar_config)], {'num_proc': 4})
    initialized = torch.load(args.drafter_checkpoint, map_location='cpu', weights_only=False)
    draft_runtime = drafter_runtime_settings(initialized)
    draft_config = get_config('DIFF_GRM', 'AmazonReviews2023CleanGR', files,
                              dict(draft_runtime, num_proc=4))
    accelerator = Accelerator()
    for config in (ar_config, draft_config):
        config.update(device='cuda', accelerator=accelerator, use_ddp=False)
    dataset = get_dataset('AmazonReviews2023CleanGR')(draft_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    if tokenizer.sid_prefix_strategy != 'none' or ar_config['n_digit'] != 4:
        raise ValueError('expected same-view OPQ4 without latent tokens')
    catalog = torch.tensor(catalog_codes(tokenizer, ar_config['codebook_size']),
                           dtype=torch.long, device='cuda')
    if catalog.unique(dim=0).shape != catalog.shape:
        raise ValueError('concrete catalog SIDs must be injective')
    raw = dataset.split()
    splits = args.splits.split(',')
    raw = {split: raw[split] for split in splits}
    if args.limit:
        raw = {s: d.select(range(min(args.limit, len(d)))) for s, d in raw.items()}
    expected = ar_config['expected_protocol']
    if not args.limit:
        for split, data in raw.items():
            if len(data) != expected[split]:
                raise ValueError(f'{split}: {len(data)} != {expected[split]}')
    if catalog.shape[0] != expected['items']:
        raise ValueError('catalog size differs from the fixed protocol')
    tokenized = tokenizer.tokenize(raw)
    drafter = EncoderOnlyFourHeadDrafter(draft_config, dataset, tokenizer).cuda().eval()
    selector = PairwisePathSelector(4, ar_config['codebook_size'], ar_config['n_embd'], rank=51).cuda().eval()
    drafter.load_state_dict(initialized['model'])
    selector.load_state_dict(initialized['selector'])
    ar = AR_GRM(ar_config, dataset, tokenizer).cuda().eval()
    ar.load_state_dict(torch.load(args.ar_checkpoint, map_location='cpu', weights_only=True))
    np.save(root / 'catalog.npy', catalog.cpu().numpy().astype(np.int16))
    serial_config = {k: v for k, v in ar_config.items() if k != 'accelerator'}
    # Config values originate in YAML; preserve only values accepted by JSON.
    serial_config = json.loads(json.dumps(serial_config, default=str))
    inputs = [args.common_config, args.sid_config, args.ar_config,
              args.drafter_checkpoint, args.ar_checkpoint, Path(ar_config['sid_override_path'])]
    split_files = {'train': 'train.jsonl', 'val': 'valid.jsonl', 'test': 'test.jsonl'}
    inputs += [Path(ar_config['data_dir']) / ar_config['splits_dir'] / split_files[s] for s in splits]
    inputs.append(Path(ar_config['data_dir']) / ar_config['item_vocab_file'])
    meta = {
        'created_utc': stamp(), 'ar_config': serial_config,
        'tokenizer': {'vocab_size': tokenizer.vocab_size, 'sid_offset': tokenizer.sid_offset,
                      'sid_prefix_strategy': 'none'},
        'ar_checkpoint': str(args.ar_checkpoint.resolve()),
        'ar_sha256': sha256(args.ar_checkpoint),
        'drafter_checkpoint': str(args.drafter_checkpoint.resolve()),
        'drafter_runtime': draft_runtime,
        'inputs': {str(p.resolve()): sha256(p) for p in inputs},
        'k': args.k, 'limit': args.limit, 'splits': {s: len(d) for s, d in tokenized.items()},
        'train_candidates': 'positive at column 0 + highest 71 non-target draft items',
        'eval_candidates': 'unaltered top72; never inject the target',
    }
    atomic_json(root / 'metadata.json', meta)
    for split, data in tokenized.items():
        directory = root / split
        directory.mkdir()
        shapes = {'history': ((len(data), ar_config['max_history_len'], 4), 'int16'),
                  'targets': ((len(data),), 'int32'),
                  'candidate_rows': ((len(data), args.k), 'int32'),
                  'proposal_scores': ((len(data), args.k), 'float32')}
        if split != 'train':
            shapes['ar_scores'] = ((len(data), args.k), 'float32')
        arrays = {name: np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+',
                                                dtype=dtype, shape=shape)
                  for name, (shape, dtype) in shapes.items()}
        # Match the reference validation batch size for baseline parity.
        loader = DataLoader(data, batch_size=args.prepare_batch_size if split == 'train' else 32, shuffle=False,
                            collate_fn=tokenizer.collate_fn[split])
        offset = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'prepare {split}'):
                targets = batch['decoder_labels' if split == 'train' else 'labels'].cuda()
                target_rows = code_rows(targets, catalog, ar.codebook_size)
                scores = one_pass_outputs(drafter, batch, catalog, selector)[0]
                if split == 'train':
                    positive = scores.gather(1, target_rows[:, None])
                    scores.scatter_(1, target_rows[:, None], float('-inf'))
                    negatives, rows = scores.topk(args.k - 1, dim=1)
                    rows = torch.cat([target_rows[:, None], rows], 1)
                    proposed = torch.cat([positive, negatives], 1)
                else:
                    proposed, rows = scores.topk(args.k, dim=1)
                    ar_scores = ar.score_candidate_paths(batch, catalog[rows], chunk_size=16)
                    arrays['ar_scores'][offset:offset + len(targets)] = ar_scores.cpu().numpy()
                end = offset + len(targets)
                arrays['history'][offset:end] = batch['history_sid'].cpu().numpy()
                arrays['targets'][offset:end] = target_rows.cpu().numpy()
                arrays['candidate_rows'][offset:end] = rows.cpu().numpy()
                arrays['proposal_scores'][offset:end] = proposed.cpu().numpy()
                offset = end
                if offset % (args.prepare_batch_size * 20) == 0:
                    atomic_json(root / 'progress.json', {'stage': f'prepare/{split}',
                                'examples': offset, 'total': len(data), 'updated_utc': stamp()})
        for array in arrays.values():
            array.flush()
        if split != 'train':
            result, ranks = score_report(arrays['proposal_scores'], arrays['ar_scores'],
                                        arrays['candidate_rows'], arrays['targets'])
            if args.reference_result and split == 'val' and not args.limit:
                expected_metrics = json.loads(args.reference_result.read_text())['validation_with_verifier']
                differences = {}
                for alpha in ALPHAS:
                    tag = f'{alpha:g}'.replace('.', 'p')
                    for name, value in result['by_alpha'][str(alpha)].items():
                        key = f'fused_a{tag}_{name}'
                        differences[key] = abs(value - expected_metrics[key])
                parity = {'max_absolute_error': max(differences.values()), 'differences': differences}
                atomic_json(root / 'parity.json', parity)
                if parity['max_absolute_error'] > 1e-6:
                    raise RuntimeError(f'baseline parity failed: {parity}')
            atomic_json(root / f'baseline_{split}.json', result)
            np.savez_compressed(root / f'baseline_{split}_ranks.npz', **ranks)
    atomic_json(root / 'ready.json', {'completed_utc': stamp(),
                'metadata_sha256': sha256(root / 'metadata.json'),
                'array_hashes': {str(p.relative_to(root)): sha256(p)
                                 for p in sorted(root.rglob('*.npy'))}})


@torch.no_grad()
def evaluate(ranker, arrays, catalog, batch_size, chunk_size):
    ranker.eval()
    output = np.empty(arrays['candidate_rows'].shape, dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(output), batch_size):
        idx = slice(start, start + batch_size)
        history = torch.tensor(np.asarray(arrays['history'][idx]), device='cuda', dtype=torch.long)
        rows = torch.tensor(np.asarray(arrays['candidate_rows'][idx]), device='cuda', dtype=torch.long)
        scores, _ = ranker(history, catalog[rows], chunk_size)
        output[idx] = scores.cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError('non-finite evaluation score')
    report, ranks = score_report(arrays['proposal_scores'], output,
                                arrays['candidate_rows'], arrays['targets'])
    report['evaluation_seconds'] = time.perf_counter() - started
    return report, ranks


def train(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'progress.json').exists() or (root / 'result.json').exists():
        raise FileExistsError('existing training artifacts; use a fresh output directory')
    ready = json.loads((args.cache / 'ready.json').read_text())
    if sha256(args.cache / 'metadata.json') != ready['metadata_sha256']:
        raise RuntimeError('candidate cache metadata changed')
    for name, expected_digest in ready['array_hashes'].items():
        if sha256(args.cache / name) != expected_digest:
            raise RuntimeError(f'candidate cache changed: {name}')
    train_arrays = load_arrays(args.cache, 'train')
    val_arrays = load_arrays(args.cache, 'val')
    catalog = torch.tensor(np.load(args.cache / 'catalog.npy'), device='cuda', dtype=torch.long)
    ar, meta = load_ar(args.cache, 'cuda')
    torch.manual_seed(args.seed + 10)
    torch.cuda.manual_seed_all(args.seed + 10)
    ranker = CandidateRankVerifier(ar, args.attention).cuda()
    optimizer = torch.optim.AdamW([
        {'params': ranker.head.parameters(), 'lr': args.head_lr},
        {'params': ranker.ar.decoder_blocks.parameters(), 'lr': args.decoder_lr},
    ], weight_decay=args.weight_decay)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(n_train=len(train_arrays['targets']), n_val=len(val_arrays['targets']),
                    parameters=sum(p.numel() for p in ranker.parameters()),
                    head_parameters=sum(p.numel() for p in ranker.head.parameters()),
                    decoder_parameters=sum(p.numel() for p in ranker.ar.decoder_blocks.parameters()),
                    initial_head_sha256=hashlib.sha256(b''.join(
                        p.detach().cpu().numpy().tobytes() for p in ranker.head.parameters())).hexdigest(),
                    cache_metadata_sha256=ready['metadata_sha256'])
    atomic_json(root / 'settings.json', settings)
    history = []
    best = float('-inf')
    best_epoch = 0
    # Initial evaluation is diagnostic only; a random head is not an AR baseline.
    initial, _ = evaluate(ranker, val_arrays, catalog, args.eval_batch_size, args.chunk_size)
    atomic_json(root / 'initial_validation.json', initial)
    for epoch in range(1, args.epochs + 1):
        ranker.set_decoder_trainable(epoch > args.warmup_epochs)
        ranker.train()
        random.seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        torch.cuda.manual_seed_all(args.seed + epoch)
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_arrays['targets']))
        total_batches = (len(order) + args.batch_size - 1) // args.batch_size
        losses, rank_losses, aux_losses = [], [], []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, start in enumerate(tqdm(range(0, len(order), args.batch_size), desc=f'{args.attention} epoch {epoch}')):
            idx = order[start:start + args.batch_size]
            h = torch.tensor(np.asarray(train_arrays['history'][idx]), device='cuda', dtype=torch.long)
            rows = torch.tensor(np.asarray(train_arrays['candidate_rows'][idx]), device='cuda', dtype=torch.long)
            targets = torch.tensor(np.asarray(train_arrays['targets'][idx]), device='cuda', dtype=torch.long)
            if not torch.equal(rows[:, 0], targets):
                raise RuntimeError('positive-first training cache invariant broken')
            scores, auxiliary = ranker(h, catalog[rows], args.chunk_size,
                                       targets=catalog[targets] if epoch > args.warmup_epochs else None)
            rank_loss = F.cross_entropy(scores, torch.zeros(len(idx), device='cuda', dtype=torch.long))
            aux_loss = auxiliary if auxiliary is not None else rank_loss.new_zeros(())
            loss = rank_loss + args.token_weight * aux_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            # Weight the final, possibly short accumulation group by examples.
            group_start = (step // args.accumulate) * args.accumulate * args.batch_size
            group_examples = min(args.batch_size * args.accumulate, len(order) - group_start)
            (loss * len(idx) / group_examples).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == total_batches:
                torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            rank_losses.append(float(rank_loss.detach()))
            aux_losses.append(float(aux_loss.detach()))
            if step % 100 == 0:
                atomic_json(root / 'progress.json', {'state': 'training', 'epoch': epoch,
                            'batch': step + 1, 'total_batches': total_batches,
                            'loss': losses[-1], 'updated_utc': stamp(), 'pid': os.getpid()})
        report, ranks = evaluate(ranker, val_arrays, catalog, args.eval_batch_size, args.chunk_size)
        record = {'epoch': epoch, 'decoder_adapted': ranker.adapt_decoder,
                  'loss': float(np.mean(losses)), 'rank_loss': float(np.mean(rank_losses)),
                  'token_loss': float(np.mean(aux_losses)), 'validation': report,
                  'peak_cuda_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
                  'epoch_seconds': time.perf_counter() - started}
        history.append(record)
        if report['selected']['ndcg@10'] > best:
            best = report['selected']['ndcg@10']
            best_epoch = epoch
            temporary = root / 'best.tmp.pt'
            torch.save({'model': ranker.state_dict(), 'epoch': epoch, 'attention': args.attention,
                        'cache_metadata_sha256': ready['metadata_sha256']}, temporary)
            temporary.replace(root / 'best.pt')
            np.savez_compressed(root / 'validation_ranks.npz', **ranks)
        atomic_json(root / 'history.json', history)
        atomic_json(root / 'progress.json', {'state': 'epoch_complete', 'epoch': epoch,
                    'best_epoch': best_epoch, 'validation': report, 'updated_utc': stamp(), 'pid': os.getpid()})
        print(json.dumps(record), flush=True)
    # Reload and reproduce the selected checkpoint, rather than reporting the last epoch.
    saved = torch.load(root / 'best.pt', map_location='cpu', weights_only=True)
    ranker.load_state_dict(saved['model'])
    selected, ranks = evaluate(ranker, val_arrays, catalog, args.eval_batch_size, args.chunk_size)
    expected = history[best_epoch - 1]['validation']
    if max(abs(selected['selected'][k] - expected['selected'][k]) for k in selected['selected']) > 1e-7:
        raise RuntimeError('selected checkpoint failed reload parity')
    np.savez_compressed(root / 'validation_ranks.npz', **ranks)
    atomic_json(root / 'result.json', {'state': 'complete', 'attention': args.attention,
                'best_epoch': best_epoch, 'validation': selected,
                'baseline': json.loads((args.cache / 'baseline_val.json').read_text()),
                'settings': settings, 'completed_utc': stamp()})
    atomic_json(root / 'progress.json', {'state': 'complete', 'best_epoch': best_epoch,
                'updated_utc': stamp(), 'pid': os.getpid()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train'])
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--common-config', type=Path)
    parser.add_argument('--sid-config', type=Path)
    parser.add_argument('--ar-config', type=Path)
    parser.add_argument('--drafter-checkpoint', type=Path)
    parser.add_argument('--ar-checkpoint', type=Path)
    parser.add_argument('--reference-result', type=Path)
    parser.add_argument('--splits', default='train,val')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--prepare-batch-size', type=int, default=128)
    parser.add_argument('--k', type=int, default=72)
    parser.add_argument('--attention', choices=['causal', 'bidirectional'], default='causal')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--head-lr', type=float, default=3e-4)
    parser.add_argument('--decoder-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--token-weight', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.k != 72:
        raise ValueError('this controlled suite fixes candidate K=72')
    if min(args.batch_size, args.accumulate, args.eval_batch_size, args.chunk_size,
           args.prepare_batch_size) < 1 or args.limit < 0:
        raise ValueError('batch/chunk sizes must be positive; limit must be non-negative')
    if args.mode == 'train' and (args.output is None or not 0 <= args.warmup_epochs < args.epochs):
        raise ValueError('training needs an output and at least one decoder adaptation epoch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU fallback')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mode == 'prepare':
        prepare(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
