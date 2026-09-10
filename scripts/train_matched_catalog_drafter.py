#!/usr/bin/env python
"""Strict atomic/unary/pairwise catalog-drafter comparison.

All arms share a frozen AR history encoder, the same pooling trunk, data order,
full legal-catalog cross entropy, optimization budget, candidate budgets, and
the same frozen AR path verifier.  Only the catalog representation changes.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.matched_catalog_drafter import MatchedCatalogDrafter
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.parallel_drafter import code_rows
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--sid-config', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument(
        '--representation', choices=('atomic', 'unary', 'pairwise'), required=True
    )
    parser.add_argument('--pair-rank', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--proposal-ks', default='72,128')
    parser.add_argument('--fusion-alphas', default='0,0.25,0.5,0.75,0.9,1')
    parser.add_argument('--max-train-examples', type=int, default=None)
    parser.add_argument('--max-val-examples', type=int, default=None)
    parser.add_argument('--max-test-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def limit_dataset(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    return dataset.select(range(int(maximum)))


def normalize_scores(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def rank_statistics(ranked_codes, labels, cutoff):
    cutoff = min(int(cutoff), ranked_codes.shape[1])
    matches = ranked_codes[:, :cutoff].eq(labels[:, None, :]).all(dim=-1)
    positions = torch.arange(cutoff, device=ranked_codes.device)[None]
    first = torch.where(matches, positions, cutoff).min(dim=1).values
    hits = first.lt(cutoff)
    zeros = torch.zeros_like(first, dtype=torch.float)
    return {
        'recall': hits.float(),
        'mrr': torch.where(hits, 1.0 / (first.float() + 1.0), zeros),
        'ndcg': torch.where(hits, 1.0 / torch.log2(first.float() + 2.0), zeros),
    }


@torch.no_grad()
def encode_history(ar_model, batch):
    return ar_model(batch, return_loss=False).hidden_states


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    drafter,
    ar_model,
    loader,
    catalog,
    proposal_ks,
    fusion_alphas=(),
    verify=False,
    description='evaluation',
):
    drafter.eval()
    ar_model.eval()
    aggregates = {}
    timing = {
        'history_encoder_seconds': 0.0,
        'catalog_scoring_seconds': 0.0,
    }
    for keep in proposal_ks:
        if verify:
            timing[f'ar_verify_k{keep}_seconds'] = 0.0
    n_examples = 0

    for batch in tqdm(loader, desc=description):
        history_sid = batch['history_sid'].to(catalog.device)
        labels = batch['labels'].to(catalog.device)

        _sync(catalog.device)
        started = time.perf_counter()
        encoder_hidden = encode_history(ar_model, batch)
        _sync(catalog.device)
        timing['history_encoder_seconds'] += time.perf_counter() - started

        started = time.perf_counter()
        scores = drafter(encoder_hidden, history_sid)
        max_keep = min(max(proposal_ks), catalog.shape[0])
        max_scores, max_rows = torch.topk(scores, k=max_keep, dim=1)
        _sync(catalog.device)
        timing['catalog_scoring_seconds'] += time.perf_counter() - started

        for requested_keep in proposal_ks:
            keep = min(int(requested_keep), catalog.shape[0])
            proposal_scores = max_scores[:, :keep]
            proposal_rows = max_rows[:, :keep]
            proposals = catalog[proposal_rows]
            stats = rank_statistics(proposals, labels, keep)
            for name, values in stats.items():
                aggregates.setdefault(
                    f'candidate_{name}@{keep}', []
                ).extend(values.cpu().tolist())

            if not verify:
                continue
            _sync(catalog.device)
            started = time.perf_counter()
            ar_scores = ar_model.score_candidate_paths(batch, proposals)
            _sync(catalog.device)
            timing[f'ar_verify_k{keep}_seconds'] += time.perf_counter() - started

            ar_order = ar_scores.argsort(dim=1, descending=True)
            ar_ranked = proposals.gather(
                1, ar_order.unsqueeze(-1).expand_as(proposals)
            )
            for cutoff in (5, 10):
                stats = rank_statistics(ar_ranked, labels, cutoff)
                for name, values in stats.items():
                    aggregates.setdefault(
                        f'ar_k{keep}_{name}@{cutoff}', []
                    ).extend(values.cpu().tolist())

            for alpha in fusion_alphas:
                fused = (
                    (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                    + float(alpha) * normalize_scores(ar_scores)
                )
                order = fused.argsort(dim=1, descending=True)
                ranked = proposals.gather(
                    1, order.unsqueeze(-1).expand_as(proposals)
                )
                tag = f'{float(alpha):g}'.replace('.', 'p')
                for cutoff in (5, 10):
                    stats = rank_statistics(ranked, labels, cutoff)
                    for name, values in stats.items():
                        aggregates.setdefault(
                            f'fused_k{keep}_a{tag}_{name}@{cutoff}', []
                        ).extend(values.cpu().tolist())
        n_examples += labels.shape[0]

    result = {name: float(np.mean(values)) for name, values in aggregates.items()}
    result['n_examples'] = n_examples
    result['timing'] = dict(timing)
    for name, seconds in timing.items():
        result['timing'][name.replace('_seconds', '_milliseconds_per_example')] = (
            1000.0 * seconds / max(n_examples, 1)
        )
    for keep in proposal_ks:
        if verify:
            result['timing'][f'end_to_end_k{keep}_milliseconds_per_example'] = (
                1000.0
                * (
                    timing['history_encoder_seconds']
                    + timing['catalog_scoring_seconds']
                    + timing[f'ar_verify_k{keep}_seconds']
                )
                / max(n_examples, 1)
            )
    return result


def main():
    args = parse_args()
    proposal_ks = sorted({int(value) for value in args.proposal_ks.split(',')})
    fusion_alphas = [float(value) for value in args.fusion_alphas.split(',')]
    if not proposal_ks or min(proposal_ks) <= 0:
        raise ValueError('--proposal-ks must contain positive integers')
    if not fusion_alphas or any(not 0 <= alpha <= 1 for alpha in fusion_alphas):
        raise ValueError('--fusion-alphas must contain values in [0,1]')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()
    config = get_config(
        'AR_GRM',
        args.dataset,
        [args.common_config, args.ar_config, args.sid_config],
        {
            'train_batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_ddp': False,
            'accelerator': accelerator,
        },
    )
    device = torch.device(config['device'])
    dataset = get_dataset(args.dataset)(config)
    tokenizer = AR_GRMTokenizer(config, dataset)
    tokenized = tokenizer.tokenize(dataset.split())
    raw_catalog = catalog_codes(tokenizer, config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('matched experiment requires collision-free item SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    train_data = limit_dataset(tokenized['train'], args.max_train_examples)
    val_data = limit_dataset(tokenized['val'], args.max_val_examples)
    test_data = limit_dataset(tokenized['test'], args.max_test_examples)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=tokenizer.collate_fn['train'],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )

    ar_model = AR_GRM(config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    ar_model.eval()
    for parameter in ar_model.parameters():
        parameter.requires_grad_(False)

    drafter = MatchedCatalogDrafter(
        ar_model,
        catalog,
        representation=args.representation,
        pair_rank=args.pair_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(
        drafter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    selection_k = min(proposal_ks)
    selection_key = f'candidate_recall@{selection_k}'
    best_value = float('-inf')
    best_path = output_dir / 'best.pt'
    for epoch in range(1, args.epochs + 1):
        drafter.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            history_sid = batch['history_sid'].to(device)
            targets = batch['decoder_labels'].to(device)
            target_rows = code_rows(targets, catalog, config['codebook_size'])
            with torch.no_grad():
                encoder_hidden = encode_history(ar_model, batch)
            scores = drafter(encoder_hidden, history_sid)
            loss = F.cross_entropy(scores, target_rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = evaluate(
            drafter,
            ar_model,
            val_loader,
            catalog,
            proposal_ks,
            description=f'validation epoch {epoch}',
        )
        validation.update(epoch=epoch, train_loss=float(np.mean(losses)))
        history.append(validation)
        print(json.dumps(validation, sort_keys=True), flush=True)
        if validation[selection_key] > best_value:
            best_value = validation[selection_key]
            torch.save(
                {
                    'drafter': drafter.state_dict(),
                    'args': vars(args),
                    'validation': validation,
                    'parameter_report': drafter.parameter_report(),
                },
                best_path,
            )

    best = torch.load(best_path, map_location=device)
    drafter.load_state_dict(best['drafter'])
    validation_verified = evaluate(
        drafter,
        ar_model,
        val_loader,
        catalog,
        proposal_ks,
        fusion_alphas=fusion_alphas,
        verify=True,
        description='validation frozen-AR verification',
    )
    selected_alphas = {}
    for keep in proposal_ks:
        alpha_values = {}
        for alpha in fusion_alphas:
            tag = f'{alpha:g}'.replace('.', 'p')
            alpha_values[alpha] = validation_verified[
                f'fused_k{keep}_a{tag}_ndcg@10'
            ]
        selected_alphas[str(keep)] = max(alpha_values, key=alpha_values.get)

    test = evaluate(
        drafter,
        ar_model,
        test_loader,
        catalog,
        proposal_ks,
        fusion_alphas=fusion_alphas,
        verify=True,
        description='test frozen-AR verification',
    )
    selected_test = {}
    for keep in proposal_ks:
        alpha = selected_alphas[str(keep)]
        tag = f'{alpha:g}'.replace('.', 'p')
        selected_test[str(keep)] = {
            'alpha': alpha,
            'candidate_recall': test[f'candidate_recall@{keep}'],
            'candidate_mrr': test[f'candidate_mrr@{keep}'],
            'ar_ndcg@10': test[f'ar_k{keep}_ndcg@10'],
            'ar_recall@10': test[f'ar_k{keep}_recall@10'],
            'fusion_ndcg@10': test[f'fused_k{keep}_a{tag}_ndcg@10'],
            'fusion_recall@10': test[f'fused_k{keep}_a{tag}_recall@10'],
            'milliseconds_per_example': test['timing'][
                f'end_to_end_k{keep}_milliseconds_per_example'
            ],
        }

    report = {
        'method': f'matched_catalog_{args.representation}',
        'protocol': vars(args),
        'catalog_items': int(catalog.shape[0]),
        'parameter_report': drafter.parameter_report(),
        'frozen_ar_parameters': sum(p.numel() for p in ar_model.parameters()),
        'best_validation': best['validation'],
        'validation_with_verifier': validation_verified,
        'selected_fusion_alphas': selected_alphas,
        'selected_test': selected_test,
        'history': history,
        'test': test,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(selected_test, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
