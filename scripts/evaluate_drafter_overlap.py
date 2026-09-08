#!/usr/bin/env python
"""Measure candidate overlap and union oracle for MIPS/semantic drafters."""

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
from genrec.models.AR_GRM.ann_drafter import ExactMIPSDrafter
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.parallel_drafter import (
    PairwisePathSelector,
    code_rows,
)
from genrec.utils import get_config, get_dataset
from scripts.train_parallel_opq_drafter import one_pass_outputs, ranking_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--mips-checkpoint', required=True)
    parser.add_argument('--semantic-checkpoint', required=True)
    parser.add_argument('--proposal-k', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--max-test-examples', type=int, default=None)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def make_config(model, dataset, files, accelerator, overrides=None):
    config = get_config(model, dataset, files, overrides or {})
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator
    return config


def unique_union(left, right):
    """Return padded row-wise unions plus a validity mask."""
    batch, width = left.shape
    rows = torch.empty(
        batch, 2 * width, dtype=torch.long, device=left.device
    )
    valid = torch.zeros(
        batch, 2 * width, dtype=torch.bool, device=left.device
    )
    sizes = []
    for batch_idx in range(batch):
        seen = set()
        merged = []
        for value in torch.cat([left[batch_idx], right[batch_idx]]).tolist():
            if value not in seen:
                seen.add(value)
                merged.append(value)
        size = len(merged)
        sizes.append(size)
        rows[batch_idx, :size] = torch.tensor(
            merged, dtype=torch.long, device=left.device
        )
        rows[batch_idx, size:] = merged[0]
        valid[batch_idx, :size] = True
    return rows, valid, torch.tensor(sizes, device=left.device)


@torch.no_grad()
def main():
    args = parse_args()
    accelerator = Accelerator()
    diffusion_config = make_config(
        'DIFF_GRM', args.dataset,
        [args.common_config, args.diffusion_config], accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    ar_config = make_config(
        'AR_GRM', args.dataset,
        [args.common_config, args.ar_config], accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    device = torch.device(diffusion_config['device'])
    dataset = get_dataset(args.dataset)(diffusion_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    tokenized = tokenizer.tokenize(dataset.split())
    test_data = tokenized['test']
    if args.max_test_examples is not None:
        test_data = test_data.select(range(min(len(test_data), args.max_test_examples)))
    loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('overlap evaluation requires collision-free item SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    ar_model.eval()

    mips_payload = torch.load(args.mips_checkpoint, map_location=device)
    mips = ExactMIPSDrafter(ar_model, catalog, 'id').to(device)
    mips.load_state_dict(mips_payload['drafter'])
    mips.eval()

    semantic_payload = torch.load(args.semantic_checkpoint, map_location=device)
    semantic_model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    semantic_model.load_state_dict(semantic_payload['model'])
    semantic_args = semantic_payload.get('args', {})
    selector = PairwisePathSelector(
        semantic_model.n_digit,
        semantic_model.codebook_size,
        semantic_model.n_embd,
        rank=int(semantic_args.get('pair_rank', 32)),
        triple_rank=(
            int(semantic_args.get('triple_rank', 0))
            if semantic_args.get('variant') == 'triple' else 0
        ),
    ).to(device)
    selector.load_state_dict(semantic_payload['selector'])
    semantic_model.eval()
    selector.eval()

    aggregates = {}
    started = time.perf_counter()
    for batch in tqdm(loader, desc='MIPS/semantic overlap'):
        labels = batch['labels'].to(device)
        target_rows = code_rows(labels, catalog, ar_config['codebook_size'])
        history_sid = batch['history_sid'].to(device)
        encoder_hidden = ar_model(batch, return_loss=False).hidden_states
        mips_scores = mips(encoder_hidden, history_sid)
        mips_rows = torch.topk(mips_scores, k=args.proposal_k, dim=1).indices

        semantic_scores, _, _, _, _ = one_pass_outputs(
            semantic_model, batch, catalog, selector
        )
        semantic_rows = torch.topk(
            semantic_scores, k=args.proposal_k, dim=1
        ).indices
        mips_hit = mips_rows.eq(target_rows[:, None]).any(dim=1)
        semantic_hit = semantic_rows.eq(target_rows[:, None]).any(dim=1)
        intersection_sizes = (
            mips_rows[:, :, None].eq(semantic_rows[:, None, :]).any(dim=2).sum(dim=1)
        )
        union_rows, union_valid, union_sizes = unique_union(
            mips_rows, semantic_rows
        )
        union_hit = union_rows.eq(target_rows[:, None]).logical_and(
            union_valid
        ).any(dim=1)

        batch_metrics = {
            f'mips_candidate_recall@{args.proposal_k}': mips_hit.float(),
            f'semantic_candidate_recall@{args.proposal_k}': semantic_hit.float(),
            f'union_oracle_recall@{2 * args.proposal_k}': union_hit.float(),
            'target_in_both_rate': (mips_hit & semantic_hit).float(),
            'target_mips_only_rate': (mips_hit & ~semantic_hit).float(),
            'target_semantic_only_rate': (~mips_hit & semantic_hit).float(),
            'target_in_neither_rate': (~mips_hit & ~semantic_hit).float(),
            'candidate_intersection_size': intersection_sizes.float(),
            'candidate_union_size': union_sizes.float(),
            'candidate_jaccard': intersection_sizes.float() / union_sizes.float(),
        }

        union_codes = catalog[union_rows]
        ar_scores = ar_model.score_candidate_paths(batch, union_codes)
        ar_scores = ar_scores.masked_fill(~union_valid, float('-inf'))
        order = ar_scores.argsort(dim=1, descending=True)
        ranked = union_codes.gather(1, order.unsqueeze(-1).expand_as(union_codes))
        for name, values in ranking_metrics(ranked, labels).items():
            batch_metrics[f'union_ar_verified_{name}'] = values
        for name, values in batch_metrics.items():
            aggregates.setdefault(name, []).extend(values.cpu().tolist())

    elapsed = time.perf_counter() - started
    result = {name: float(np.mean(values)) for name, values in aggregates.items()}
    result.update(
        n_examples=len(test_data),
        elapsed_seconds=elapsed,
        milliseconds_per_example=1000.0 * elapsed / max(len(test_data), 1),
        protocol=vars(args),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
