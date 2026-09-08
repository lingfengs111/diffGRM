#!/usr/bin/env python
"""Check exactness and latency of cached AR candidate-path verification."""

import argparse
import json
from pathlib import Path
import sys
import time

from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--candidate-k', type=int, default=72)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed_score(model, batch, candidates, cached, chunk_size, iterations, device):
    synchronize(device)
    started = time.perf_counter()
    output = None
    with torch.no_grad():
        for _ in range(iterations):
            output = model.score_candidate_paths(
                batch,
                candidates,
                chunk_size=chunk_size,
                use_cached_history=cached,
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    return output, elapsed / iterations


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    accelerator = Accelerator()
    config = get_config(
        'AR_GRM',
        args.dataset,
        [args.common_config, args.ar_config],
        {'eval_batch_size': args.batch_size, 'use_ddp': False},
    )
    config['accelerator'] = accelerator
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(config['device'])

    dataset = get_dataset(args.dataset)(config)
    tokenizer = AR_GRMTokenizer(config, dataset)
    tokenized = tokenizer.tokenize(dataset.split())
    loader = DataLoader(
        tokenized[args.split],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn[args.split],
    )
    batch = next(iter(loader))

    model = AR_GRM(config, dataset, tokenizer).to(device)
    model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    model.eval()

    catalog = torch.as_tensor(
        catalog_codes(tokenizer, config['codebook_size']),
        dtype=torch.long,
        device=device,
    )
    batch_size = batch['history_sid'].shape[0]
    offsets = torch.arange(batch_size, device=device)[:, None]
    rows = (
        torch.arange(args.candidate_k, device=device)[None, :] + offsets
    ) % catalog.shape[0]
    candidates = catalog[rows]

    with torch.no_grad():
        for cached in (False, True):
            for _ in range(args.warmup):
                model.score_candidate_paths(
                    batch,
                    candidates,
                    chunk_size=args.chunk_size,
                    use_cached_history=cached,
                )

    reference, uncached_seconds = timed_score(
        model, batch, candidates, False, args.chunk_size,
        args.iterations, device,
    )
    cached, cached_seconds = timed_score(
        model, batch, candidates, True, args.chunk_size,
        args.iterations, device,
    )
    difference = (reference - cached).abs()
    report = {
        'batch_size': batch_size,
        'candidate_k': args.candidate_k,
        'chunk_size': args.chunk_size,
        'iterations': args.iterations,
        'max_absolute_score_difference': float(difference.max().item()),
        'mean_absolute_score_difference': float(difference.mean().item()),
        'identical_candidate_order': bool(
            torch.equal(
                reference.argsort(dim=1, descending=True),
                cached.argsort(dim=1, descending=True),
            )
        ),
        'uncached_milliseconds_per_batch': 1000.0 * uncached_seconds,
        'cached_milliseconds_per_batch': 1000.0 * cached_seconds,
        'speedup': uncached_seconds / cached_seconds,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )


if __name__ == '__main__':
    main()
