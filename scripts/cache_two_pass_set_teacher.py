#!/usr/bin/env python
"""Cache a two-pass typed-set teacher distribution for one-pass distillation."""

import argparse
import json
from pathlib import Path
import sys
import time

from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector
from genrec.utils import get_config, get_dataset
from scripts.train_parallel_opq_drafter import (
    encode_history,
    limit_dataset,
    make_config,
    one_pass_outputs,
    two_pass_scores,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--sid-config', default=None)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--teacher-checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--branches', type=int, default=16)
    parser.add_argument('--branch-chunk', type=int, default=None)
    parser.add_argument('--first-weight', type=float, required=True)
    parser.add_argument('--preserve-first', action='store_true')
    parser.add_argument('--teacher-topk', type=int, default=72)
    parser.add_argument('--max-train-examples', type=int, default=None)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def selector_from_payload(model, payload, device):
    saved = payload.get('args', {})
    variant = saved.get('variant', 'pairwise')
    if variant == 'unary':
        return None
    selector = PairwisePathSelector(
        model.n_digit,
        model.codebook_size,
        model.n_embd,
        rank=int(saved.get('pair_rank', 32)),
        triple_rank=(
            int(saved.get('triple_rank', 0)) if variant == 'triple' else 0
        ),
    ).to(device)
    selector.load_state_dict(payload['selector'])
    return selector


@torch.no_grad()
def main():
    args = parse_args()
    if not 0.0 <= args.first_weight <= 1.0:
        raise ValueError('--first-weight must lie in [0,1]')
    accelerator = Accelerator()
    files = [args.common_config]
    if args.sid_config:
        files.append(args.sid_config)
    config = make_config(
        'DIFF_GRM', args.dataset,
        files + [args.diffusion_config], accelerator,
        {'eval_batch_size': args.batch_size},
    )
    ar_config = make_config(
        'AR_GRM', args.dataset,
        files + [args.ar_config], accelerator,
        {'eval_batch_size': args.batch_size},
    )
    device = torch.device(config['device'])
    dataset = get_dataset(args.dataset)(config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    raw = dataset.split()
    raw['train'] = limit_dataset(raw['train'], args.max_train_examples)
    tokenized = tokenizer.tokenize({'train': raw['train']})['train']
    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['train'],
    )
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('teacher caching requires collision-free SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    payload = torch.load(args.teacher_checkpoint, map_location=device)
    model = DIFF_GRM(config, dataset, tokenizer).to(device)
    model.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))
    model.load_state_dict(payload['model'])
    selector = selector_from_payload(model, payload, device)
    model.eval()
    if selector is not None:
        selector.eval()

    keep = min(int(args.teacher_topk), catalog.shape[0])
    rows_cache = torch.empty(len(tokenized), keep, dtype=torch.int32)
    scores_cache = torch.empty(len(tokenized), keep, dtype=torch.float16)
    route_cache = torch.empty(len(tokenized), dtype=torch.uint8)
    branch_cache = torch.empty(
        len(tokenized), min(args.branches, model.codebook_size), dtype=torch.int16
    )
    offset = 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc='cache two-pass set teacher'):
        encoder_hidden = encode_history(model, batch)
        first_scores, first_logits, _, _, _ = one_pass_outputs(
            model, batch, catalog, selector, encoder_hidden=encoder_hidden
        )
        fused, reveal_digit, branch_values, _ = two_pass_scores(
            model,
            encoder_hidden,
            catalog,
            selector,
            first_scores,
            first_logits,
            args.branches,
            [args.first_weight],
            branch_chunk=args.branch_chunk,
            preserve_first=args.preserve_first,
        )
        teacher_scores, teacher_rows = torch.topk(
            fused[args.first_weight], k=keep, dim=1
        )
        stop = offset + teacher_rows.shape[0]
        rows_cache[offset:stop] = teacher_rows.to(torch.int32).cpu()
        scores_cache[offset:stop] = teacher_scores.to(torch.float16).cpu()
        route_cache[offset:stop] = reveal_digit.to(torch.uint8).cpu()
        branch_cache[offset:stop] = branch_values.to(torch.int16).cpu()
        offset = stop

    elapsed = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'candidate_rows': rows_cache,
            'candidate_scores': scores_cache,
            'reveal_digit': route_cache,
            'branch_values': branch_cache,
            'protocol': vars(args),
            'n_examples': len(tokenized),
            'catalog_items': int(catalog.shape[0]),
            'elapsed_seconds': elapsed,
        },
        output,
    )
    summary = {
        'output': str(output),
        'n_examples': len(tokenized),
        'teacher_topk': keep,
        'elapsed_seconds': elapsed,
        'milliseconds_per_example': 1000.0 * elapsed / max(len(tokenized), 1),
    }
    output.with_suffix('.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
