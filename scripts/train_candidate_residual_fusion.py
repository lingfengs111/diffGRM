#!/usr/bin/env python
"""Learn tiny candidate-level fusion rules on validation proposals only.

Inputs are frozen per-candidate scores from the one-pass drafter and AR
verifier.  No recommender parameter is updated.  Model selection happens on a
deterministic validation holdout, after which the selected epoch count is
retrained on the complete validation split and evaluated once on test.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-artifact', required=True)
    parser.add_argument('--test-artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def load(path):
    path = Path(path).expanduser().resolve()
    with np.load(path / 'examples.npz') as handle:
        return {key: handle[key] for key in handle.files}


def normalize_rows(value):
    value = np.asarray(value, dtype=np.float32)
    return (value - value.mean(axis=1, keepdims=True)) / np.maximum(
        value.std(axis=1, keepdims=True), 1e-6
    )


def build_features(data, context_stats=None):
    required = {
        'proposal_item_ids', 'target_item_id', 'unary_candidate_scores',
        'pairwise_candidate_scores', 'ar_candidate_token_scores',
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f'artifact lacks candidate decomposition: {missing}')
    unary = normalize_rows(data['unary_candidate_scores'])
    pairwise = normalize_rows(data['pairwise_candidate_scores'])
    ar_tokens = data['ar_candidate_token_scores'].astype(np.float32)
    components = [
        normalize_rows(data['drafter_candidate_scores']),
        normalize_rows(data['ar_candidate_scores']),
        unary,
        pairwise,
    ]
    components.extend(
        normalize_rows(ar_tokens[:, :, digit])
        for digit in range(ar_tokens.shape[2])
    )
    features = np.stack(components, axis=2).astype(np.float32)

    context_names = [
        'drafter_ar_score_correlation', 'drafter_candidate_entropy',
        'ar_candidate_entropy', 'drafter_ar_top10_overlap',
        'drafter_fusion_top10_overlap', 'ar_fusion_top10_overlap',
    ]
    context = np.column_stack([data[name] for name in context_names]).astype(np.float32)
    if context_stats is None:
        mean = context.mean(axis=0, keepdims=True)
        std = context.std(axis=0, keepdims=True)
        context_stats = (mean, std)
    mean, std = context_stats
    context = (context - mean) / np.maximum(std, 1e-6)

    matches = data['proposal_item_ids'] == data['target_item_id'][:, None]
    present = matches.any(axis=1)
    target_position = matches.argmax(axis=1).astype(np.int64)
    return features, context, present, target_position, context_stats


def target_ranks(scores, present, target_position):
    # All inputs here are continuous full-path scores. Counting strictly
    # greater values is equivalent to sorting while making a fine alpha sweep
    # roughly two orders of magnitude cheaper.
    target_score = scores[np.arange(len(scores)), target_position]
    rank = 1 + (scores > target_score[:, None]).sum(axis=1)
    return np.where(present, rank, 0).astype(np.int16)


def ndcg_recall(scores, present, target_position):
    rank = target_ranks(scores, present, target_position)
    hit5 = (rank > 0) & (rank <= 5)
    hit10 = (rank > 0) & (rank <= 10)
    ndcg5 = np.zeros(len(rank), dtype=np.float64)
    ndcg10 = np.zeros(len(rank), dtype=np.float64)
    ndcg5[hit5] = 1.0 / np.log2(rank[hit5] + 1.0)
    ndcg10[hit10] = 1.0 / np.log2(rank[hit10] + 1.0)
    return {
        'recall@5': float(hit5.mean()),
        'ndcg@5': float(ndcg5.mean()),
        'recall@10': float(hit10.mean()),
        'ndcg@10': float(ndcg10.mean()),
    }


class SimplexLinear(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        initial = torch.full((n_features,), 1e-4)
        initial[0] = 0.25
        initial[1] = 0.75
        initial /= initial.sum()
        self.raw_weights = nn.Parameter(initial.log())

    def forward(self, features, context):
        del context
        return (features * self.raw_weights.softmax(dim=0)).sum(dim=-1)

    def describe(self):
        return {'weights': self.raw_weights.softmax(dim=0).detach().cpu().tolist()}


class UnconstrainedLinear(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_features) / n_features)

    def forward(self, features, context):
        del context
        return (features * self.weights).sum(dim=-1)

    def describe(self):
        return {'weights': self.weights.detach().cpu().tolist()}


class ResidualLinear(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.residual = nn.Parameter(torch.zeros(n_features))

    def forward(self, features, context):
        del context
        base = 0.25 * features[:, :, 0] + 0.75 * features[:, :, 1]
        return base + (features * self.residual).sum(dim=-1)

    def describe(self):
        return {'residual_weights': self.residual.detach().cpu().tolist()}


class AdaptiveAlpha(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.base_logit = float(np.log(0.75 / 0.25))
        self.gate = nn.Sequential(
            nn.Linear(context_dim, 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features, context):
        alpha = torch.sigmoid(self.base_logit + self.gate(context)).squeeze(-1)
        return (
            (1.0 - alpha[:, None]) * features[:, :, 0]
            + alpha[:, None] * features[:, :, 1]
        )

    def describe(self):
        return {}


class ResidualMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, context):
        del context
        base = 0.25 * features[:, :, 0] + 0.75 * features[:, :, 1]
        return base + self.network(features).squeeze(-1)

    def describe(self):
        return {}


class AdaptiveSimplex(nn.Module):
    def __init__(self, n_features, context_dim):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(n_features))
        self.gate = nn.Sequential(
            nn.Linear(context_dim, 16), nn.GELU(), nn.Linear(16, n_features)
        )

    def forward(self, features, context):
        weights = (self.base[None] + self.gate(context)).softmax(dim=-1)
        return (features * weights[:, None]).sum(dim=-1)

    def describe(self):
        return {'base_weights': self.base.softmax(dim=0).detach().cpu().tolist()}


class CandidateMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 16), nn.GELU(), nn.Linear(16, 1)
        )

    def forward(self, features, context):
        del context
        return self.network(features).squeeze(-1)

    def describe(self):
        return {}


def predict(model, features, context, device, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            chunks.append(model(
                torch.from_numpy(features[start:end]).to(device),
                torch.from_numpy(context[start:end]).to(device),
            ).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_epochs(
    model, features, context, present, target_position, rows, epochs,
    batch_size, device, lr, seed,
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_rows = np.asarray(rows)[present[rows]]
    generator = np.random.default_rng(seed)
    last_loss = None
    for _ in range(epochs):
        order = generator.permutation(train_rows)
        losses = []
        for start in range(0, len(order), batch_size):
            batch_rows = order[start:start + batch_size]
            logits = model(
                torch.from_numpy(features[batch_rows]).to(device),
                torch.from_numpy(context[batch_rows]).to(device),
            )
            targets = torch.from_numpy(target_position[batch_rows]).to(device)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
    return last_loss


def fit_variant(
    name, factory, lr, val, test, device, epochs, patience, batch_size, seed,
):
    val_features, val_context, val_present, val_target = val
    test_features, test_context, test_present, test_target = test
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(val_features))
    boundary = int(0.8 * len(shuffled))
    train_rows, holdout_rows = shuffled[:boundary], shuffled[boundary:]

    torch.manual_seed(seed)
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    usable_train = train_rows[val_present[train_rows]]
    initial_scores = predict(
        model, val_features[holdout_rows], val_context[holdout_rows],
        device, batch_size,
    )
    initial_metrics = ndcg_recall(
        initial_scores, val_present[holdout_rows], val_target[holdout_rows]
    )
    best_epoch = 0
    best_ndcg = initial_metrics['ndcg@10']
    no_improve = 0
    generator = np.random.default_rng(seed)
    history = [{'epoch': 0, 'loss': None, **initial_metrics}]
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(usable_train)
        losses = []
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            logits = model(
                torch.from_numpy(val_features[rows]).to(device),
                torch.from_numpy(val_context[rows]).to(device),
            )
            targets = torch.from_numpy(val_target[rows]).to(device)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        holdout_scores = predict(
            model, val_features[holdout_rows], val_context[holdout_rows],
            device, batch_size,
        )
        metrics = ndcg_recall(
            holdout_scores, val_present[holdout_rows], val_target[holdout_rows]
        )
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), **metrics})
        if metrics['ndcg@10'] > best_ndcg + 1e-10:
            best_ndcg = metrics['ndcg@10']
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Retrain from the same initialization on all validation proposals for the
    # holdout-selected number of epochs.
    torch.manual_seed(seed)
    final_model = factory().to(device)
    final_loss = train_epochs(
        final_model, val_features, val_context, val_present, val_target,
        np.arange(len(val_features)), best_epoch, batch_size, device, lr, seed,
    )
    scores = predict(final_model, test_features, test_context, device, batch_size)
    result = {
        'variant': name,
        'parameters': int(sum(p.numel() for p in final_model.parameters())),
        'selected_epochs': best_epoch,
        'holdout_best_ndcg@10': best_ndcg,
        'final_validation_loss': final_loss,
        'test': ndcg_recall(scores, test_present, test_target),
        'model': final_model.describe(),
        'selection_history': history,
    }
    return result, target_ranks(scores, test_present, test_target)


def markdown_table(rows):
    columns = ['variant', 'parameters', 'selected_epochs', 'recall@5', 'ndcg@5', 'recall@10', 'ndcg@10']
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] + ['---:' for _ in columns[1:]]) + ' |',
    ]
    for row in rows:
        flat = {**row, **row.get('test', {})}
        values = []
        for key in columns:
            value = flat.get(key, '')
            if isinstance(value, float):
                value = f'{value:.6f}'
            values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    args = parse_args()
    raw_val = load(args.validation_artifact)
    raw_test = load(args.test_artifact)
    val_features, val_context, val_present, val_target, stats = build_features(raw_val)
    test_features, test_context, test_present, test_target, _ = build_features(raw_test, stats)
    val = (val_features, val_context, val_present, val_target)
    test = (test_features, test_context, test_present, test_target)
    device = torch.device(args.device)
    n_features = val_features.shape[-1]
    context_dim = val_context.shape[-1]

    baseline_specs = [
        ('frozen drafter', raw_test['drafter_candidate_scores']),
        ('frozen AR', raw_test['ar_candidate_scores']),
        ('fixed alpha=0.75 fusion', raw_test['fusion_candidate_scores']),
    ]
    rank_arrays = {}
    fixed_rows = []
    for name, scores in baseline_specs:
        fixed_rows.append(
            {'variant': name, 'parameters': 0, 'selected_epochs': 0,
             'test': ndcg_recall(scores, test_present, test_target)}
        )
        rank_arrays[name.replace(' ', '_').replace('=', '')] = target_ranks(
            scores, test_present, test_target
        )

    val_drafter = normalize_rows(raw_val['drafter_candidate_scores'])
    val_ar = normalize_rows(raw_val['ar_candidate_scores'])
    test_drafter = normalize_rows(raw_test['drafter_candidate_scores'])
    test_ar = normalize_rows(raw_test['ar_candidate_scores'])
    best_alpha = None
    best_alpha_ndcg = float('-inf')
    for alpha in np.linspace(0.0, 1.0, 201):
        metrics = ndcg_recall(
            (1.0 - alpha) * val_drafter + alpha * val_ar,
            val_present, val_target,
        )
        if metrics['ndcg@10'] > best_alpha_ndcg:
            best_alpha_ndcg = metrics['ndcg@10']
            best_alpha = float(alpha)
    fine_scores = (1.0 - best_alpha) * test_drafter + best_alpha * test_ar
    fixed_rows.append({
        'variant': 'validation fine-grid alpha',
        'parameters': 0,
        'selected_epochs': 0,
        'selected_alpha': best_alpha,
        'test': ndcg_recall(fine_scores, test_present, test_target),
    })
    rank_arrays['validation_fine_grid_alpha'] = target_ranks(
        fine_scores, test_present, test_target
    )
    specifications = [
        ('fixed-start simplex', lambda: SimplexLinear(n_features), 5e-3),
        ('fixed-start residual linear', lambda: ResidualLinear(n_features), 2e-3),
        ('adaptive alpha', lambda: AdaptiveAlpha(context_dim), 1e-3),
        ('fixed-start residual MLP', lambda: ResidualMLP(n_features), 1e-3),
    ]
    learned = []
    for index, (name, factory, lr) in enumerate(specifications):
        row, rank = fit_variant(
            name, factory, lr, val, test, device, args.epochs,
            args.patience, args.batch_size, args.seed + index,
        )
        learned.append(row)
        rank_arrays[name.replace(' ', '_')] = rank
        print(f'[candidate fusion] complete: {name}', flush=True)
    result = {
        'feature_order': [
            'drafter_total', 'ar_total', 'unary', 'pairwise',
            'ar_c0', 'ar_c1', 'ar_c2', 'ar_c3',
        ],
        'validation_candidate_recall': float(val_present.mean()),
        'test_candidate_recall': float(test_present.mean()),
        'baselines': fixed_rows,
        'learned': learned,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    np.savez_compressed(output_dir / 'test_ranks.npz', **rank_arrays)
    report = '\n'.join([
        '# Candidate residual fusion', '',
        f"Feature order: `{result['feature_order']}`", '',
        markdown_table(fixed_rows + learned), '',
    ])
    (output_dir / 'REPORT.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
