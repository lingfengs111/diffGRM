#!/usr/bin/env python3
"""Evaluate a drafter and AR verifier that use different SID tokenizers.

The legal catalog row is the only bridge between the two views: the drafter
proposes item rows in its own SID space, and the same rows are translated to
the verifier's SID space before teacher-forced AR scoring.  No SID coordinate
is compared or copied across tokenizers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
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
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.encoder_head_drafter import EncoderOnlyFourHeadDrafter
from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector, code_rows
from genrec.utils import get_config, get_dataset
from scripts.train_parallel_opq_drafter import (
    encode_history,
    limit_dataset,
    normalize_scores,
    one_pass_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--drafter-sid-config', required=True)
    parser.add_argument('--verifier-sid-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--max-val-examples', type=int, default=None)
    parser.add_argument('--max-test-examples', type=int, default=None)
    parser.add_argument('--fusion-alphas', default='0,0.1,0.25,0.5,0.75,0.9,1')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def make_config(model, dataset, files, accelerator, overrides=None):
    config = get_config(model, dataset, files, overrides or {})
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator
    return config


def row_metrics(ranked_rows, target_rows, cutoffs=(5, 10)):
    matches = ranked_rows.eq(target_rows[:, None])
    output = {}
    positions = torch.arange(ranked_rows.shape[1], device=ranked_rows.device)[None]
    first = torch.where(matches, positions, ranked_rows.shape[1]).min(dim=1).values
    for requested in cutoffs:
        cutoff = min(int(requested), ranked_rows.shape[1])
        hits = matches[:, :cutoff].any(dim=1)
        output[f'recall@{cutoff}'] = hits.float()
        output[f'ndcg@{cutoff}'] = torch.where(
            hits,
            torch.log2(first.float() + 2.0).reciprocal(),
            torch.zeros_like(first, dtype=torch.float),
        )
    return output


@torch.no_grad()
def evaluate(
    model,
    selector,
    ar_model,
    drafter_loader,
    verifier_loader,
    drafter_catalog,
    verifier_catalog,
    proposal_k,
    alphas,
    description,
):
    model.eval()
    selector.eval()
    ar_model.eval()
    aggregates = {}
    examples = 0
    started = time.perf_counter()
    paired = zip(drafter_loader, verifier_loader, strict=True)
    for drafter_batch, verifier_batch in tqdm(
        paired, total=len(drafter_loader), desc=description
    ):
        drafter_labels = drafter_batch['labels'].to(drafter_catalog.device)
        verifier_labels = verifier_batch['labels'].to(verifier_catalog.device)
        target_rows = code_rows(
            drafter_labels,
            drafter_catalog,
            model.codebook_size,
            identity_start_digit=0,
        )
        verifier_target_rows = code_rows(
            verifier_labels,
            verifier_catalog,
            ar_model.codebook_size,
            identity_start_digit=0,
        )
        if not torch.equal(target_rows, verifier_target_rows):
            raise RuntimeError(
                'paired tokenizers disagree on catalog row ordering; '
                'dual-view evaluation would be invalid'
            )

        hidden = encode_history(model, drafter_batch, None)
        full_scores, _, _, _, _ = one_pass_outputs(
            model,
            drafter_batch,
            drafter_catalog,
            selector,
            encoder_hidden=hidden,
        )
        keep = min(int(proposal_k), int(drafter_catalog.shape[0]))
        proposal_scores, proposal_rows = torch.topk(full_scores, k=keep, dim=1)
        verifier_paths = verifier_catalog[proposal_rows]
        ar_scores = ar_model.score_candidate_paths(
            verifier_batch, verifier_paths
        )

        drafter_order = proposal_scores.argsort(dim=1, descending=True)
        drafter_rows = proposal_rows.gather(1, drafter_order)
        ar_order = ar_scores.argsort(dim=1, descending=True)
        ar_rows = proposal_rows.gather(1, ar_order)
        for prefix, ranked in (('drafter', drafter_rows), ('ar_verified', ar_rows)):
            for name, values in row_metrics(ranked, target_rows, (5, 10, keep)).items():
                aggregates.setdefault(f'{prefix}_{name}', []).extend(values.cpu().tolist())

        for alpha in alphas:
            fused = (
                (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                + float(alpha) * normalize_scores(ar_scores)
            )
            order = fused.argsort(dim=1, descending=True)
            ranked = proposal_rows.gather(1, order)
            tag = f'{float(alpha):g}'.replace('.', 'p')
            for name, values in row_metrics(ranked, target_rows, (5, 10)).items():
                aggregates.setdefault(f'fused_a{tag}_{name}', []).extend(values.cpu().tolist())
        examples += int(target_rows.shape[0])

    elapsed = time.perf_counter() - started
    result = {key: float(np.mean(value)) for key, value in aggregates.items()}
    result.update(
        n_examples=examples,
        elapsed_seconds=elapsed,
        milliseconds_per_example=1000.0 * elapsed / max(examples, 1),
    )
    return result


def main():
    args = parse_args()
    if args.output.exists():
        print(f'already complete: {args.output}', flush=True)
        return
    alphas = [float(value) for value in args.fusion_alphas.split(',')]
    if not alphas or any(not 0.0 <= value <= 1.0 for value in alphas):
        raise ValueError('fusion alphas must lie in [0,1]')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator()
    overrides = {'eval_batch_size': args.eval_batch_size}
    drafter_config = make_config(
        'DIFF_GRM', args.dataset,
        [args.common_config, args.drafter_sid_config, args.diffusion_config],
        accelerator, overrides,
    )
    drafter_tokenizer_config = make_config(
        'AR_GRM', args.dataset,
        [args.common_config, args.drafter_sid_config, args.ar_config],
        accelerator, overrides,
    )
    verifier_config = make_config(
        'AR_GRM', args.dataset,
        [args.common_config, args.verifier_sid_config, args.ar_config],
        accelerator, overrides,
    )
    device = torch.device(drafter_config['device'])
    drafter_dataset = get_dataset(args.dataset)(drafter_config)
    verifier_dataset = get_dataset(args.dataset)(verifier_config)
    drafter_tokenizer = AR_GRMTokenizer(
        drafter_tokenizer_config, drafter_dataset
    )
    verifier_tokenizer = AR_GRMTokenizer(verifier_config, verifier_dataset)

    drafter_splits = drafter_dataset.split()
    verifier_splits = verifier_dataset.split()
    for split in ('val', 'test'):
        if len(drafter_splits[split]) != len(verifier_splits[split]):
            raise RuntimeError(f'{split} sizes differ across tokenizer views')
    for splits in (drafter_splits, verifier_splits):
        # This script never trains; avoiding 2x full train tokenization makes
        # the cross-view control cheaper without changing val/test examples.
        splits['train'] = splits['train'].select(range(0))
        splits['val'] = limit_dataset(splits['val'], args.max_val_examples)
        splits['test'] = limit_dataset(splits['test'], args.max_test_examples)
    drafter_tokenized = drafter_tokenizer.tokenize(drafter_splits)
    verifier_tokenized = verifier_tokenizer.tokenize(verifier_splits)
    drafter_catalog = torch.as_tensor(
        catalog_codes(drafter_tokenizer, drafter_config['codebook_size']),
        dtype=torch.long,
        device=device,
    )
    verifier_catalog = torch.as_tensor(
        catalog_codes(verifier_tokenizer, verifier_config['codebook_size']),
        dtype=torch.long,
        device=device,
    )
    if drafter_catalog.shape[0] != verifier_catalog.shape[0]:
        raise RuntimeError('catalog sizes differ across tokenizer views')
    if np.unique(drafter_catalog.cpu().numpy(), axis=0).shape[0] != drafter_catalog.shape[0]:
        raise RuntimeError('drafter catalog contains SID collisions')
    if np.unique(verifier_catalog.cpu().numpy(), axis=0).shape[0] != verifier_catalog.shape[0]:
        raise RuntimeError('verifier catalog contains SID collisions')

    loaders = {}
    for name, tokenizer, tokenized in (
        ('drafter', drafter_tokenizer, drafter_tokenized),
        ('verifier', verifier_tokenizer, verifier_tokenized),
    ):
        loaders[name] = {
            split: DataLoader(
                tokenized[split],
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=tokenizer.collate_fn[split],
            )
            for split in ('val', 'test')
        }

    checkpoint = torch.load(args.drafter_checkpoint, map_location=device)
    checkpoint_args = checkpoint.get('args', {})
    if checkpoint_args.get('backbone_architecture', 'encoder_four_head') != 'encoder_four_head':
        raise ValueError('dual-view evaluator currently expects encoder_four_head')
    drafter_config['encoder_head_n_layer'] = int(
        checkpoint_args.get('encoder_head_n_layer', 4)
    )
    # These are behavioral settings, not state-dict tensors. The canonical
    # trainer always writes them into the runtime config, so omitting them can
    # silently change the unary/pairwise score scale after an exact checkpoint
    # load (notably temperature 0.07 -> 1.0).
    drafter_config['encoder_head_normalize_logits'] = (
        checkpoint_args.get('training_objective', 'catalog_plus_token')
        == 'mtp_only'
    )
    drafter_config['encoder_head_logit_temperature'] = float(
        checkpoint_args.get('mtp_temperature', 0.07)
    )
    drafter_config.update(
        history_head=checkpoint_args.get('history_head', 'pooled'),
        n_interests=int(checkpoint_args.get('n_interests', 1)),
        interest_temperature=float(
            checkpoint_args.get('interest_temperature', 1.0)
        ),
    )
    model = EncoderOnlyFourHeadDrafter(
        drafter_config, drafter_dataset, drafter_tokenizer
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    selector_state = checkpoint.get('selector')
    if selector_state is None:
        raise ValueError('dual-view control requires a pairwise selector checkpoint')
    selector = PairwisePathSelector(
        model.n_digit,
        model.codebook_size,
        model.n_embd,
        rank=int(checkpoint_args.get('pair_rank', 51)),
    ).to(device)
    selector.load_state_dict(selector_state)
    ar_model = AR_GRM(
        verifier_config, verifier_dataset, verifier_tokenizer
    ).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))

    validation = evaluate(
        model, selector, ar_model,
        loaders['drafter']['val'], loaders['verifier']['val'],
        drafter_catalog, verifier_catalog,
        args.proposal_k, alphas, 'dual-view validation',
    )
    selection_cutoff = min(10, args.proposal_k, int(drafter_catalog.shape[0]))
    alpha_scores = {
        alpha: validation[
            f"fused_a{float(alpha):g}_ndcg@{selection_cutoff}".replace('.', 'p')
        ]
        for alpha in alphas
    }
    selected_alpha = max(alpha_scores, key=alpha_scores.get)
    test = evaluate(
        model, selector, ar_model,
        loaders['drafter']['test'], loaders['verifier']['test'],
        drafter_catalog, verifier_catalog,
        args.proposal_k, [selected_alpha], 'dual-view test',
    )
    report = {
        'protocol': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'catalog_items': int(drafter_catalog.shape[0]),
        'drafter_digits': int(drafter_catalog.shape[1]),
        'verifier_digits': int(verifier_catalog.shape[1]),
        'selected_fusion_alpha': selected_alpha,
        'validation': validation,
        'test': test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
