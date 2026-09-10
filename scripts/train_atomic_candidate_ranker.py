#!/usr/bin/env python
"""Train a completely SID-free atomic retriever and candidate ranker.

Histories, targets, catalog parameters, proposals, and final ranking all use
ordinary item IDs.  This is the control that answers whether semantic-ID
factorization is necessary anywhere in the two-stage system.
"""

import argparse
from functools import partial
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

from genrec.models.AR_GRM.atomic_recommender import (
    AtomicCandidateRanker,
    AtomicSequentialRetriever,
)
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--domain-config', required=True)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-inner', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--retriever-epochs', type=int, default=20)
    parser.add_argument('--ranker-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--retriever-lr', type=float, default=3e-4)
    parser.add_argument('--ranker-lr', type=float, default=1e-3)
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


def collate_atomic(batch, item2id, max_history_len):
    histories = []
    targets = []
    for example in batch:
        sequence = example['item_seq']
        if len(sequence) < 2:
            raise ValueError('atomic training example has no history')
        history = [item2id[item] for item in sequence[:-1]][-max_history_len:]
        history = history + [0] * (max_history_len - len(history))
        histories.append(history)
        targets.append(item2id[sequence[-1]])
    return {
        'history_ids': torch.tensor(histories, dtype=torch.long),
        'target_ids': torch.tensor(targets, dtype=torch.long),
    }


def normalize_scores(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def score_tag(value):
    return f'{float(value):g}'.replace('.', 'p')


def rank_statistics(ranked_ids, target_ids, cutoff):
    cutoff = min(int(cutoff), ranked_ids.shape[1])
    matches = ranked_ids[:, :cutoff].eq(target_ids[:, None])
    positions = torch.arange(cutoff, device=ranked_ids.device)[None]
    first = torch.where(matches, positions, cutoff).min(dim=1).values
    hits = first.lt(cutoff)
    zeros = torch.zeros_like(first, dtype=torch.float)
    return {
        'recall': hits.float(),
        'mrr': torch.where(hits, 1.0 / (first.float() + 1.0), zeros),
        'ndcg': torch.where(hits, 1.0 / torch.log2(first.float() + 2.0), zeros),
    }


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    retriever,
    loader,
    proposal_ks,
    ranker=None,
    fusion_alphas=(),
    description='evaluation',
):
    retriever.eval()
    if ranker is not None:
        ranker.eval()
    device = next(retriever.parameters()).device
    aggregates = {}
    timing = {'retriever_seconds': 0.0}
    if ranker is not None:
        for keep in proposal_ks:
            timing[f'ranker_k{keep}_seconds'] = 0.0
    n_examples = 0

    for batch in tqdm(loader, desc=description):
        history = batch['history_ids'].to(device)
        targets = batch['target_ids'].to(device)
        _sync(device)
        started = time.perf_counter()
        scores, query = retriever(history)
        max_keep = min(max(proposal_ks), scores.shape[1])
        max_scores, max_rows = torch.topk(scores, k=max_keep, dim=1)
        _sync(device)
        timing['retriever_seconds'] += time.perf_counter() - started

        for requested_keep in proposal_ks:
            keep = min(int(requested_keep), scores.shape[1])
            candidate_rows = max_rows[:, :keep]
            candidate_ids = candidate_rows + 1
            candidate_scores = max_scores[:, :keep]
            stats = rank_statistics(candidate_ids, targets, keep)
            for name, values in stats.items():
                aggregates.setdefault(
                    f'candidate_{name}@{keep}', []
                ).extend(values.cpu().tolist())

            for cutoff in (5, 10):
                stats = rank_statistics(candidate_ids, targets, cutoff)
                for name, values in stats.items():
                    aggregates.setdefault(
                        f'retriever_{name}@{cutoff}', []
                    ).extend(values.cpu().tolist())
            if ranker is None:
                continue

            _sync(device)
            started = time.perf_counter()
            vectors = retriever.item_embedding(candidate_ids)
            ranker_scores = ranker(query, vectors)
            _sync(device)
            timing[f'ranker_k{keep}_seconds'] += time.perf_counter() - started

            order = ranker_scores.argsort(dim=1, descending=True)
            ranked = candidate_ids.gather(1, order)
            for cutoff in (5, 10):
                stats = rank_statistics(ranked, targets, cutoff)
                for name, values in stats.items():
                    aggregates.setdefault(
                        f'ranker_k{keep}_{name}@{cutoff}', []
                    ).extend(values.cpu().tolist())

            for alpha in fusion_alphas:
                fused = (
                    (1.0 - float(alpha)) * normalize_scores(candidate_scores)
                    + float(alpha) * normalize_scores(ranker_scores)
                )
                order = fused.argsort(dim=1, descending=True)
                ranked = candidate_ids.gather(1, order)
                tag = score_tag(alpha)
                for cutoff in (5, 10):
                    stats = rank_statistics(ranked, targets, cutoff)
                    for name, values in stats.items():
                        aggregates.setdefault(
                            f'fused_k{keep}_a{tag}_{name}@{cutoff}', []
                        ).extend(values.cpu().tolist())
        n_examples += targets.shape[0]

    result = {name: float(np.mean(values)) for name, values in aggregates.items()}
    result['n_examples'] = n_examples
    result['timing'] = dict(timing)
    for name, seconds in timing.items():
        result['timing'][name.replace('_seconds', '_milliseconds_per_example')] = (
            1000.0 * seconds / max(n_examples, 1)
        )
    for keep in proposal_ks:
        if ranker is not None:
            result['timing'][f'end_to_end_k{keep}_milliseconds_per_example'] = (
                1000.0
                * (timing['retriever_seconds'] + timing[f'ranker_k{keep}_seconds'])
                / max(n_examples, 1)
            )
    return result


def force_target_into_candidates(scores, candidate_rows, target_rows):
    candidate_scores = scores.gather(1, candidate_rows)
    matches = candidate_rows.eq(target_rows[:, None])
    absent = ~matches.any(dim=1)
    if absent.any():
        candidate_rows = candidate_rows.clone()
        candidate_scores = candidate_scores.clone()
        candidate_rows[absent, -1] = target_rows[absent]
        candidate_scores[absent, -1] = scores[absent, target_rows[absent]]
    target_positions = candidate_rows.eq(target_rows[:, None]).float().argmax(dim=1)
    return candidate_rows, candidate_scores, target_positions


def main():
    args = parse_args()
    proposal_ks = sorted({int(value) for value in args.proposal_ks.split(',')})
    fusion_alphas = [float(value) for value in args.fusion_alphas.split(',')]
    if not proposal_ks or min(proposal_ks) <= 0:
        raise ValueError('--proposal-ks must contain positive integers')
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
        [args.common_config, args.domain_config],
        {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_ddp': False,
            'accelerator': accelerator,
        },
    )
    device = torch.device(config['device'])
    dataset = get_dataset(args.dataset)(config)
    splits = dataset.split()
    collate = partial(
        collate_atomic,
        item2id=dataset.item2id,
        max_history_len=int(config['max_history_len']),
    )
    train_data = limit_dataset(splits['train'], args.max_train_examples)
    val_data = limit_dataset(splits['val'], args.max_val_examples)
    test_data = limit_dataset(splits['test'], args.max_test_examples)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    retriever = AtomicSequentialRetriever(
        n_items=dataset.n_items - 1,
        max_history_len=int(config['max_history_len']),
        hidden_dim=args.hidden_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        dropout=args.dropout,
    ).to(device)
    retriever_optimizer = torch.optim.AdamW(
        retriever.parameters(),
        lr=args.retriever_lr,
        weight_decay=args.weight_decay,
    )
    retriever_history = []
    best_recall = float('-inf')
    retriever_path = output_dir / 'best_retriever.pt'
    selection_k = min(proposal_ks)
    for epoch in range(1, args.retriever_epochs + 1):
        retriever.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'retriever epoch {epoch}'):
            history = batch['history_ids'].to(device)
            target_rows = batch['target_ids'].to(device) - 1
            retriever_optimizer.zero_grad(set_to_none=True)
            scores, _ = retriever(history)
            loss = F.cross_entropy(scores, target_rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
            retriever_optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(
            retriever,
            val_loader,
            proposal_ks,
            description=f'retriever validation epoch {epoch}',
        )
        validation.update(epoch=epoch, train_loss=float(np.mean(losses)))
        retriever_history.append(validation)
        print(json.dumps(validation, sort_keys=True), flush=True)
        value = validation[f'candidate_recall@{selection_k}']
        if value > best_recall:
            best_recall = value
            torch.save(
                {'retriever': retriever.state_dict(), 'validation': validation},
                retriever_path,
            )

    best_retriever = torch.load(retriever_path, map_location=device)
    retriever.load_state_dict(best_retriever['retriever'])
    retriever.eval()
    for parameter in retriever.parameters():
        parameter.requires_grad_(False)

    ranker = AtomicCandidateRanker(args.hidden_dim).to(device)
    ranker_optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=args.ranker_lr, weight_decay=args.weight_decay
    )
    ranker_history = []
    best_ranker_ndcg = float('-inf')
    ranker_path = output_dir / 'best_ranker.pt'
    train_keep = max(proposal_ks)
    for epoch in range(1, args.ranker_epochs + 1):
        ranker.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'atomic ranker epoch {epoch}'):
            history = batch['history_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_rows = target_ids - 1
            with torch.no_grad():
                scores, query = retriever(history)
                candidate_rows = scores.topk(train_keep, dim=1).indices
                candidate_rows, _, target_positions = force_target_into_candidates(
                    scores, candidate_rows, target_rows
                )
                candidate_ids = candidate_rows + 1
                vectors = retriever.item_embedding(candidate_ids)
            ranker_optimizer.zero_grad(set_to_none=True)
            ranker_scores = ranker(query, vectors)
            loss = F.cross_entropy(ranker_scores, target_positions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            ranker_optimizer.step()
            losses.append(float(loss.detach()))

        validation = evaluate(
            retriever,
            val_loader,
            proposal_ks,
            ranker=ranker,
            fusion_alphas=fusion_alphas,
            description=f'atomic ranker validation epoch {epoch}',
        )
        validation.update(epoch=epoch, train_loss=float(np.mean(losses)))
        ranker_history.append(validation)
        selected_value = max(
            validation[
                f'fused_k{selection_k}_a{score_tag(alpha)}_ndcg@10'
            ]
            for alpha in fusion_alphas
        )
        print(json.dumps(validation, sort_keys=True), flush=True)
        if selected_value > best_ranker_ndcg:
            best_ranker_ndcg = selected_value
            torch.save(
                {'ranker': ranker.state_dict(), 'validation': validation},
                ranker_path,
            )

    best_ranker = torch.load(ranker_path, map_location=device)
    ranker.load_state_dict(best_ranker['ranker'])
    validation = evaluate(
        retriever,
        val_loader,
        proposal_ks,
        ranker=ranker,
        fusion_alphas=fusion_alphas,
        description='final atomic validation',
    )
    selected_alphas = {}
    for keep in proposal_ks:
        alpha_values = {}
        for alpha in fusion_alphas:
            tag = score_tag(alpha)
            alpha_values[alpha] = validation[f'fused_k{keep}_a{tag}_ndcg@10']
        selected_alphas[str(keep)] = max(alpha_values, key=alpha_values.get)

    test = evaluate(
        retriever,
        test_loader,
        proposal_ks,
        ranker=ranker,
        fusion_alphas=fusion_alphas,
        description='atomic test',
    )
    selected_test = {}
    for keep in proposal_ks:
        alpha = selected_alphas[str(keep)]
        tag = score_tag(alpha)
        selected_test[str(keep)] = {
            'alpha': alpha,
            'candidate_recall': test[f'candidate_recall@{keep}'],
            'candidate_mrr': test[f'candidate_mrr@{keep}'],
            'ranker_ndcg@10': test[f'ranker_k{keep}_ndcg@10'],
            'ranker_recall@10': test[f'ranker_k{keep}_recall@10'],
            'fusion_ndcg@10': test[f'fused_k{keep}_a{tag}_ndcg@10'],
            'fusion_recall@10': test[
                f'fused_k{keep}_a{tag}_recall@10'
            ],
            'milliseconds_per_example': test['timing'][
                f'end_to_end_k{keep}_milliseconds_per_example'
            ],
        }

    report = {
        'method': 'atomic_item_retriever_plus_atomic_candidate_ranker_no_sid',
        'protocol': vars(args),
        'catalog_items': dataset.n_items - 1,
        'retriever_parameters': sum(p.numel() for p in retriever.parameters()),
        'ranker_parameters': sum(p.numel() for p in ranker.parameters()),
        'best_retriever_validation': best_retriever['validation'],
        'best_ranker_validation': best_ranker['validation'],
        'selected_fusion_alphas': selected_alphas,
        'selected_test': selected_test,
        'retriever_history': retriever_history,
        'ranker_history': ranker_history,
        'test': test,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(selected_test, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
