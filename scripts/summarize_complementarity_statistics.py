#!/usr/bin/env python
"""Statistical analysis of a saved drafter/AR complementarity artifact.

This script is deliberately model-free: it consumes ``examples.npz`` written
by ``analyze_drafter_ar_complementarity.py``.  Consequently bootstrap,
specialization, and rank-flow analyses can be iterated without another GPU
forward pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def rank_utility(rank, metric, cutoff=10):
    rank = np.asarray(rank)
    hit = (rank > 0) & (rank <= cutoff)
    if metric == 'recall':
        return hit.astype(np.float64)
    if metric == 'ndcg':
        result = np.zeros(rank.shape, dtype=np.float64)
        result[hit] = 1.0 / np.log2(rank[hit] + 1.0)
        return result
    raise ValueError(metric)


def rank_metrics(rank):
    return {
        f'{metric}@{cutoff}': float(rank_utility(rank, metric, cutoff).mean())
        for cutoff in (5, 10)
        for metric in ('recall', 'ndcg')
    }


def candidate_target_rank(scores, candidate_ids, target_ids):
    scores = np.asarray(scores, dtype=np.float32)
    # Stable sorting is essential for partial-path scores: c0-only and c0:c1
    # deliberately create many exact ties. Candidate order is the drafter rank,
    # so stable ties mean "AR reorders only when it has evidence" rather than
    # giving every member of a tied code group the same optimistic rank.
    order = np.argsort(-scores, axis=1, kind='stable')
    ordered_ids = np.take_along_axis(candidate_ids, order, axis=1)
    matches = ordered_ids == target_ids[:, None]
    present = matches.any(axis=1)
    rank = matches.argmax(axis=1) + 1
    return np.where(present, rank, 0).astype(np.int32)


def normalize_rows(scores):
    scores = np.asarray(scores, dtype=np.float32)
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-6
    )


def path_score_ablation(data):
    required = {
        'proposal_item_ids', 'target_item_id', 'drafter_candidate_scores',
        'ar_candidate_token_scores',
    }
    if not required.issubset(data):
        return []
    candidates = data['proposal_item_ids']
    targets = data['target_item_id']
    drafter = data['drafter_candidate_scores']
    ar_tokens = data['ar_candidate_token_scores'].astype(np.float32)
    alpha = float(data['fusion_alpha'][0])
    rows = []

    variants = [('AR c0 only', ar_tokens[:, :, 0])]
    cumulative = np.cumsum(ar_tokens, axis=2)
    for end in range(1, ar_tokens.shape[2]):
        variants.append((f'AR c0:c{end}', cumulative[:, :, end]))
    variants.append(('AR late c1:c3', ar_tokens[:, :, 1:].sum(axis=2)))
    for name, scores in variants:
        rank = candidate_target_rank(scores, candidates, targets)
        row = {'scorer': name, **rank_metrics(rank)}
        fused = (
            (1.0 - alpha) * normalize_rows(drafter)
            + alpha * normalize_rows(scores)
        )
        fused_rank = candidate_target_rank(fused, candidates, targets)
        fused_metrics = rank_metrics(fused_rank)
        row.update({f'fused_{key}': value for key, value in fused_metrics.items()})
        rows.append(row)

    if 'drafter_candidate_token_scores' in data:
        drafter_tokens = data['drafter_candidate_token_scores'].astype(np.float32)
        drafter_cumulative = np.cumsum(drafter_tokens, axis=2)
        for end in range(drafter_tokens.shape[2]):
            rank = candidate_target_rank(
                drafter_cumulative[:, :, end], candidates, targets
            )
            rows.append({
                'scorer': f'drafter unary c0:c{end}',
                **rank_metrics(rank),
            })
    return rows


def paired_bootstrap(left, right, rng, samples):
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    n = len(difference)
    estimates = np.empty(samples, dtype=np.float64)
    # Chunked index generation avoids a potentially huge [samples, n] matrix.
    for start in range(0, samples, 128):
        end = min(start + 128, samples)
        indices = rng.integers(0, n, size=(end - start, n))
        estimates[start:end] = difference[indices].mean(axis=1)
    return {
        'difference': float(difference.mean()),
        'ci95_low': float(np.quantile(estimates, 0.025)),
        'ci95_high': float(np.quantile(estimates, 0.975)),
        'bootstrap_probability_gt_zero': float((estimates > 0).mean()),
    }


def mcnemar(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    left_only = int((left & ~right).sum())
    right_only = int((~left & right).sum())
    discordant = left_only + right_only
    pvalue = 1.0 if discordant == 0 else float(
        binomtest(min(left_only, right_only), discordant, 0.5).pvalue
    )
    return {
        'left_only': left_only,
        'right_only': right_only,
        'discordant': discordant,
        'exact_two_sided_p': pvalue,
    }


def standardized_mean_difference(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return None
    pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
    if pooled <= 0:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def feature_matrix(data):
    names = [
        'log1p_target_train_count',
        'log1p_last_transition_count',
        'history_length',
        'history_unique_ratio',
        'repeat_target',
        'text_last_cosine',
        'text_max_cosine',
        'sid_last_match_count',
        'sid_max_match_count',
    ]
    columns = [
        np.log1p(data['target_train_count']),
        np.log1p(data['last_transition_count']),
        data['history_length'],
        data['history_unique_ratio'],
        data['repeat_target'].astype(np.float32),
        data['text_last_cosine'],
        data['text_max_cosine'],
        data['sid_last_match_count'],
        data['sid_max_match_count'],
    ]
    return names, np.column_stack(columns).astype(np.float64)


def fit_specialization_probe(data, drafter_only, ar_only, seed):
    mask = drafter_only | ar_only
    labels = ar_only[mask].astype(np.int64)
    names, features = feature_matrix(data)
    features = features[mask]
    if len(np.unique(labels)) < 2 or min(np.bincount(labels)) < 5:
        return {'status': 'insufficient_examples'}
    folds = min(5, int(np.bincount(labels).min()))
    model = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=seed
        ),
    )
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    probability = cross_val_predict(
        model, features, labels, cv=splitter, method='predict_proba'
    )[:, 1]
    auc = float(roc_auc_score(labels, probability))
    model.fit(features, labels)
    coefficients = model.named_steps['logisticregression'].coef_[0]
    ordered = sorted(
        (
            {'feature': name, 'standardized_coefficient': float(value)}
            for name, value in zip(names, coefficients)
        ),
        key=lambda row: abs(row['standardized_coefficient']),
        reverse=True,
    )
    return {
        'status': 'ok',
        'task': 'AR-only (1) versus drafter-only (0)',
        'n_examples': int(mask.sum()),
        'ar_only_fraction': float(labels.mean()),
        'cross_validated_roc_auc': auc,
        'coefficients': ordered,
    }


def rank_bucket(rank, proposal_k):
    rank = np.asarray(rank)
    labels = np.full(rank.shape, f'>{proposal_k}', dtype=f'<U{len(str(proposal_k)) + 4}')
    labels[(rank >= 1) & (rank <= 10)] = '1-10'
    labels[(rank >= 11) & (rank <= 32)] = '11-32'
    labels[(rank >= 33) & (rank <= proposal_k)] = f'33-{proposal_k}'
    return labels


def rank_flow(data, proposal_k):
    source = rank_bucket(data['drafter_rank'], proposal_k)
    rows = []
    for label in ('1-10', '11-32', f'33-{proposal_k}', f'>{proposal_k}'):
        mask = source == label
        if not mask.any():
            continue
        rows.append({
            'drafter_rank_bucket': label,
            'count': int(mask.sum()),
            'fraction': float(mask.mean()),
            'ar_recall@10': float(
                ((data['ar_rank'][mask] > 0) & (data['ar_rank'][mask] <= 10)).mean()
            ),
            'fusion_recall@10': float(
                ((data['fusion_rank'][mask] > 0) & (data['fusion_rank'][mask] <= 10)).mean()
            ),
            'mean_ar_counterfactual_rank': float(
                data['ar_counterfactual_rank'][mask].mean()
            ),
            'mean_fusion_counterfactual_rank': float(
                data['fusion_counterfactual_rank'][mask].mean()
            ),
        })
    return rows


def markdown_table(rows, columns):
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] + ['---:' for _ in columns[1:]]) + ' |',
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, '')
            if isinstance(value, float):
                value = f'{value:.6f}'
            values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def render_report(result):
    lines = [
        '# Complementarity statistical report',
        '',
        f"- examples: {result['n_examples']:,}",
        f"- proposal K: {result['proposal_k']}",
        '',
        '## Paired metric differences',
        '',
    ]
    rows = []
    for name, value in result['paired_differences'].items():
        rows.append({
            'comparison': name,
            'difference': value['difference'],
            'ci95_low': value['ci95_low'],
            'ci95_high': value['ci95_high'],
            'P(>0)': value['bootstrap_probability_gt_zero'],
        })
    lines.extend([
        markdown_table(
            rows, ['comparison', 'difference', 'ci95_low', 'ci95_high', 'P(>0)']
        ),
        '',
        '## Recall@10 discordance',
        '',
        markdown_table(
            [dict(comparison=name, **value) for name, value in result['mcnemar'].items()],
            ['comparison', 'left_only', 'right_only', 'discordant', 'exact_two_sided_p'],
        ),
        '',
        '## Drafter-rank flow',
        '',
        markdown_table(
            result['rank_flow'],
            [
                'drafter_rank_bucket', 'count', 'fraction', 'ar_recall@10',
                'fusion_recall@10', 'mean_ar_counterfactual_rank',
                'mean_fusion_counterfactual_rank',
            ],
        ),
        '',
        '## AR-only versus drafter-only specialization',
        '',
    ])
    probe = result['specialization_probe']
    if probe['status'] == 'ok':
        lines.extend([
            f"Cross-validated ROC-AUC from model-independent attributes: "
            f"**{probe['cross_validated_roc_auc']:.4f}**.",
            '',
            markdown_table(
                probe['coefficients'], ['feature', 'standardized_coefficient']
            ),
        ])
    else:
        lines.append(probe['status'])
    lines.extend([
        '',
        'Positive coefficients indicate an association with AR-only examples; '
        'negative coefficients indicate drafter-only examples. This is '
        'descriptive evidence, not a causal interpretation.',
        '',
        '## Coordinate target-log-probability gaps',
        '',
        markdown_table(
            result['coordinate_logp'],
            ['coordinate', 'all_ar_minus_drafter', 'drafter_only_ar_minus_drafter',
             'ar_only_ar_minus_drafter', 'drafter_token_accuracy',
             'ar_teacher_forced_token_accuracy'],
        ),
    ])
    if 'pairwise_contribution' in result:
        pairwise = result['pairwise_contribution']
        lines.extend([
            '',
            '## Pairwise selector contribution',
            '',
            f"- unary-only Recall@10: {pairwise['unary_recall@10']:.6f}",
            f"- unary/drafter oracle union Recall@10: "
            f"{pairwise['unary_union_drafter_recall@10']:.6f}",
            f"- pairwise-rescued unary misses: "
            f"{pairwise['pairwise_rescued_unary_miss']:,}",
            f"- pairwise-harmed unary hits: "
            f"{pairwise['pairwise_harmed_unary_hit']:,}",
        ])
    if result.get('path_score_ablation'):
        lines.extend([
            '',
            '## Candidate-path score ablation',
            '',
            markdown_table(
                result['path_score_ablation'],
                ['scorer', 'recall@5', 'ndcg@5', 'recall@10', 'ndcg@10',
                 'fused_recall@5', 'fused_ndcg@5', 'fused_recall@10',
                 'fused_ndcg@10'],
            ),
        ])
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    with np.load(artifact_dir / 'examples.npz') as handle:
        data = {key: handle[key] for key in handle.files}
    n = len(data['drafter_rank'])
    proposal_k = int(data['proposal_k'][0])
    rng = np.random.default_rng(args.seed)

    rank_keys = {
        'drafter': 'drafter_rank',
        'candidate_ar': 'ar_rank',
        'fusion': 'fusion_rank',
    }
    if 'unary_rank' in data:
        rank_keys['unary_only'] = 'unary_rank'
    if 'standalone_ar_rank' in data:
        rank_keys['standalone_ar'] = 'standalone_ar_rank'

    comparisons = [('fusion', 'drafter'), ('fusion', 'candidate_ar')]
    if 'unary_only' in rank_keys:
        comparisons.extend([
            ('drafter', 'unary_only'),
            ('fusion', 'unary_only'),
        ])
    if 'standalone_ar' in rank_keys:
        comparisons.extend([
            ('fusion', 'standalone_ar'),
            ('candidate_ar', 'standalone_ar'),
        ])
    paired = {}
    for left, right in comparisons:
        for metric in ('recall', 'ndcg'):
            name = f'{left}_minus_{right}_{metric}@10'
            paired[name] = paired_bootstrap(
                rank_utility(data[rank_keys[left]], metric),
                rank_utility(data[rank_keys[right]], metric),
                rng,
                args.bootstrap_samples,
            )

    hit = {
        name: (data[key] > 0) & (data[key] <= 10)
        for name, key in rank_keys.items()
    }
    mcnemar_results = {
        f'{left}_vs_{right}': mcnemar(hit[left], hit[right])
        for left, right in comparisons
    }
    drafter_only = hit['drafter'] & ~hit['candidate_ar']
    ar_only = ~hit['drafter'] & hit['candidate_ar']

    names, features = feature_matrix(data)
    effects = []
    for index, name in enumerate(names):
        effects.append({
            'feature': name,
            'ar_only_minus_drafter_only_smd': standardized_mean_difference(
                features[ar_only, index], features[drafter_only, index]
            ),
        })

    coordinate_rows = []
    gap = data['target_ar_token_logp'] - data['target_drafter_token_logp']
    for coordinate in range(gap.shape[1]):
        coordinate_rows.append({
            'coordinate': coordinate,
            'all_ar_minus_drafter': float(gap[:, coordinate].mean()),
            'drafter_only_ar_minus_drafter': float(
                gap[drafter_only, coordinate].mean()
            ) if drafter_only.any() else None,
            'ar_only_ar_minus_drafter': float(
                gap[ar_only, coordinate].mean()
            ) if ar_only.any() else None,
            'drafter_token_accuracy': float(
                data['drafter_token_correct'][:, coordinate].mean()
            ) if 'drafter_token_correct' in data else None,
            'ar_teacher_forced_token_accuracy': float(
                data['ar_teacher_forced_token_correct'][:, coordinate].mean()
            ) if 'ar_teacher_forced_token_correct' in data else None,
        })

    result = {
        'artifact_dir': str(artifact_dir),
        'n_examples': n,
        'proposal_k': proposal_k,
        'paired_differences': paired,
        'mcnemar': mcnemar_results,
        'rank_flow': rank_flow(data, proposal_k),
        'specialization_effect_sizes': effects,
        'specialization_probe': fit_specialization_probe(
            data, drafter_only, ar_only, args.seed
        ),
        'coordinate_logp': coordinate_rows,
        'path_score_ablation': path_score_ablation(data),
    }
    if 'unary_only' in rank_keys:
        unary_hit = hit['unary_only']
        result['pairwise_contribution'] = {
            'unary_recall@10': float(unary_hit.mean()),
            'unary_union_drafter_recall@10': float(
                (unary_hit | hit['drafter']).mean()
            ),
            'pairwise_rescued_unary_miss': int(
                (hit['drafter'] & ~unary_hit).sum()
            ),
            'pairwise_harmed_unary_hit': int(
                (~hit['drafter'] & unary_hit).sum()
            ),
        }
    with open(artifact_dir / 'statistics.json', 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    (artifact_dir / 'STATISTICAL_REPORT.md').write_text(
        render_report(result), encoding='utf-8'
    )
    print(json.dumps({
        'artifact_dir': str(artifact_dir),
        'n_examples': n,
        'specialization_probe': result['specialization_probe'],
        'paired_differences': paired,
    }, indent=2))


if __name__ == '__main__':
    main()
