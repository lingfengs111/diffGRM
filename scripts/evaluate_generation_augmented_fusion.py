#!/usr/bin/env python3
"""Evaluate a fixed-budget union of parallel and genuine AR proposals.

The experiment keeps the final candidate budget constant.  A one-pass
structured drafter contributes ``draft_k`` items and constrained AR beam
search contributes ``ar_k`` genuinely generated items.  Duplicates are
removed and any empty slots are filled from the remaining drafter ranking.
The same cached teacher-forced AR verifier then scores all candidates.
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
    parser.add_argument('--sid-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--candidate-k', type=int, default=72)
    parser.add_argument('--draft-k', type=int, default=56)
    parser.add_argument('--ar-k', type=int, default=16)
    parser.add_argument(
        '--ar-search-beam', type=int, default=16,
        help='True constrained AR search width, separate from returned ar-k.',
    )
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


def row_metrics(ranked_rows, target_rows, cutoffs):
    matches = ranked_rows.eq(target_rows[:, None])
    positions = torch.arange(
        ranked_rows.shape[1], device=ranked_rows.device
    )[None]
    first = torch.where(matches, positions, ranked_rows.shape[1]).min(dim=1).values
    output = {}
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


def fixed_budget_union(draft_rows, generated_rows, draft_k, ar_k, candidate_k):
    """Deduplicate two ordered lists and fill every row to exactly candidate_k."""
    draft_cpu = draft_rows.detach().cpu().tolist()
    generated_cpu = generated_rows.detach().cpu().tolist()
    merged = []
    generated_unique = []
    for draft, generated in zip(draft_cpu, generated_cpu):
        selected = []
        seen = set()
        for row in draft[:draft_k]:
            if row not in seen:
                selected.append(row)
                seen.add(row)
        new_from_ar = 0
        for row in generated[:ar_k]:
            if row not in seen:
                selected.append(row)
                seen.add(row)
                new_from_ar += 1
            if len(selected) == candidate_k:
                break
        for row in draft[draft_k:]:
            if len(selected) == candidate_k:
                break
            if row not in seen:
                selected.append(row)
                seen.add(row)
        if len(selected) != candidate_k:
            raise RuntimeError(
                f'could only construct {len(selected)} unique candidates, '
                f'expected {candidate_k}'
            )
        merged.append(selected)
        generated_unique.append(new_from_ar)
    return (
        torch.as_tensor(merged, dtype=torch.long, device=draft_rows.device),
        torch.as_tensor(
            generated_unique, dtype=torch.float, device=draft_rows.device
        ),
    )


@torch.no_grad()
def evaluate(
    drafter,
    selector,
    ar_model,
    loader,
    catalog,
    candidate_k,
    draft_k,
    ar_k,
    ar_search_beam,
    alphas,
    split,
):
    drafter.eval()
    selector.eval()
    ar_model.eval()
    ar_model.config['current_split'] = split
    beam_config = ar_model.config.setdefault('ar_beam_search', {})
    beam_config['top_k_final'] = int(ar_k)
    beam_config[split] = {
        'pre_cut_num': [int(ar_search_beam)] * ar_model.n_digit,
        'beam_search_num': [int(ar_search_beam)] * ar_model.n_digit,
    }

    aggregates = {}
    examples = 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc=f'generation-augmented {split}'):
        targets = batch['labels'].to(catalog.device)
        target_rows = code_rows(
            targets, catalog, drafter.codebook_size, identity_start_digit=0
        )
        hidden = encode_history(drafter, batch, None)
        full_scores, _, _, _, _ = one_pass_outputs(
            drafter, batch, catalog, selector, encoder_hidden=hidden
        )
        # candidate_k drafter rows are sufficient for both the primary prefix
        # and deterministic fill after AR/drafter overlap is removed.
        _, draft_rows = torch.topk(full_scores, k=candidate_k, dim=1)

        generated_codes = ar_model.generate(
            batch, n_return_sequences=ar_k
        )
        generated_rows = code_rows(
            generated_codes.reshape(-1, generated_codes.shape[-1]),
            catalog,
            ar_model.codebook_size,
            identity_start_digit=0,
        ).reshape(generated_codes.shape[:2])
        union_rows, new_from_ar = fixed_budget_union(
            draft_rows, generated_rows, draft_k, ar_k, candidate_k
        )
        union_paths = catalog[union_rows]
        proposal_scores = full_scores.gather(1, union_rows)
        ar_scores = ar_model.score_candidate_paths(batch, union_paths)

        draft72_hit = draft_rows.eq(target_rows[:, None]).any(dim=1)
        draft_prefix_hit = draft_rows[:, :draft_k].eq(
            target_rows[:, None]
        ).any(dim=1)
        generated_hit = generated_rows[:, :ar_k].eq(
            target_rows[:, None]
        ).any(dim=1)
        union_hit = union_rows.eq(target_rows[:, None]).any(dim=1)
        rescued = (~draft_prefix_hit) & generated_hit
        lost_vs_draft72 = draft72_hit & (~union_hit)
        for name, values in (
            ('draft_candidate_recall', draft72_hit.float()),
            ('draft_prefix_recall', draft_prefix_hit.float()),
            ('ar_generated_recall', generated_hit.float()),
            ('union_candidate_recall', union_hit.float()),
            ('ar_rescue_rate', rescued.float()),
            ('lost_vs_draft72_rate', lost_vs_draft72.float()),
            ('new_unique_ar_candidates', new_from_ar),
        ):
            aggregates.setdefault(name, []).extend(values.cpu().tolist())

        generated_top = generated_rows[:, :min(10, ar_k)]
        for name, values in row_metrics(
            generated_top, target_rows, (5, 10)
        ).items():
            aggregates.setdefault(f'ar_generate_{name}', []).extend(
                values.cpu().tolist()
            )

        ar_order = ar_scores.argsort(dim=1, descending=True)
        ar_ranked = union_rows.gather(1, ar_order)
        for name, values in row_metrics(
            ar_ranked, target_rows, (5, 10)
        ).items():
            aggregates.setdefault(f'ar_verified_{name}', []).extend(
                values.cpu().tolist()
            )

        for alpha in alphas:
            fused = (
                (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                + float(alpha) * normalize_scores(ar_scores)
            )
            ranked = union_rows.gather(
                1, fused.argsort(dim=1, descending=True)
            )
            tag = f'{float(alpha):g}'.replace('.', 'p')
            for name, values in row_metrics(
                ranked, target_rows, (5, 10)
            ).items():
                aggregates.setdefault(f'fused_a{tag}_{name}', []).extend(
                    values.cpu().tolist()
                )
        examples += int(target_rows.shape[0])

    elapsed = time.perf_counter() - started
    result = {key: float(np.mean(values)) for key, values in aggregates.items()}
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
    if args.draft_k + args.ar_k != args.candidate_k:
        raise ValueError('fixed-budget protocol requires draft_k + ar_k == candidate_k')
    if min(args.candidate_k, args.draft_k, args.ar_k) < 1:
        raise ValueError('candidate budgets must be positive')
    alphas = [float(value) for value in args.fusion_alphas.split(',')]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator()
    files = [args.common_config, args.sid_config]
    overrides = {'eval_batch_size': args.eval_batch_size}
    drafter_config = make_config(
        'DIFF_GRM', args.dataset, files + [args.diffusion_config],
        accelerator, overrides,
    )
    ar_config = make_config(
        'AR_GRM', args.dataset, files + [args.ar_config],
        accelerator, overrides,
    )
    device = torch.device(drafter_config['device'])
    dataset = get_dataset(args.dataset)(drafter_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    splits = dataset.split()
    splits['train'] = splits['train'].select(range(0))
    splits['val'] = limit_dataset(splits['val'], args.max_val_examples)
    splits['test'] = limit_dataset(splits['test'], args.max_test_examples)
    tokenized = tokenizer.tokenize(splits)
    loaders = {
        split: DataLoader(
            tokenized[split], batch_size=args.eval_batch_size,
            shuffle=False, collate_fn=tokenizer.collate_fn[split],
        )
        for split in ('val', 'test')
    }
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise RuntimeError('generation-augmented evaluation requires unique SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    checkpoint = torch.load(args.drafter_checkpoint, map_location=device)
    checkpoint_args = checkpoint.get('args', {})
    if checkpoint_args.get('backbone_architecture') != 'encoder_four_head':
        raise ValueError('this evaluator expects an encoder_four_head drafter')
    drafter_config['encoder_head_n_layer'] = int(
        checkpoint_args.get('encoder_head_n_layer', 4)
    )
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
        interest_temperature=float(checkpoint_args.get('interest_temperature', 1.0)),
    )
    drafter = EncoderOnlyFourHeadDrafter(
        drafter_config, dataset, tokenizer
    ).to(device)
    drafter.load_state_dict(checkpoint['model'])
    selector = PairwisePathSelector(
        drafter.n_digit,
        drafter.codebook_size,
        drafter.n_embd,
        rank=int(checkpoint_args.get('pair_rank', 51)),
    ).to(device)
    selector.load_state_dict(checkpoint['selector'])
    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))

    validation = evaluate(
        drafter, selector, ar_model, loaders['val'], catalog,
        args.candidate_k, args.draft_k, args.ar_k, args.ar_search_beam,
        alphas, 'val',
    )
    alpha_scores = {
        alpha: validation[
            f"fused_a{float(alpha):g}_ndcg@10".replace('.', 'p')
        ]
        for alpha in alphas
    }
    selected_alpha = max(alpha_scores, key=alpha_scores.get)
    test = evaluate(
        drafter, selector, ar_model, loaders['test'], catalog,
        args.candidate_k, args.draft_k, args.ar_k, args.ar_search_beam,
        [selected_alpha], 'test',
    )
    report = {
        'protocol': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'selected_fusion_alpha': selected_alpha,
        'validation': validation,
        'test': test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
