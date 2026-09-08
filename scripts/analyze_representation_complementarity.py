#!/usr/bin/env python
"""Representation-level diagnosis for a drafter and an AR verifier.

The validation artifact is used to train every probe/router; the test artifact
is touched only for final evaluation.  This avoids the common but invalid
practice of fitting a sample router directly on test-set error quadrants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-artifact', required=True)
    parser.add_argument('--test-artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--probe-epochs', type=int, default=12)
    parser.add_argument('--probe-batch-size', type=int, default=4096)
    parser.add_argument('--cka-samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def load_artifact(path):
    path = Path(path).expanduser().resolve()
    with np.load(path / 'examples.npz') as handle:
        result = {key: handle[key] for key in handle.files}
    required = {
        'drafter_history_state', 'ar_history_state',
        'drafter_coordinate_states', 'target_codes',
    }
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError(f'{path} lacks representation fields: {missing}')
    return result


def centered_linear_cka(left, right, max_samples, seed):
    if len(left) != len(right):
        raise ValueError('CKA views must be row aligned')
    if len(left) > max_samples:
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(left), size=max_samples, replace=False)
        left = left[rows]
        right = right[rows]
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    cross = left.T @ right
    left_cov = left.T @ left
    right_cov = right.T @ right
    numerator = float(np.square(cross).sum())
    denominator = float(
        np.sqrt(np.square(left_cov).sum() * np.square(right_cov).sum())
    )
    return numerator / max(denominator, 1e-20)


def external_features(data):
    names = [
        'log1p_target_train_count', 'log1p_last_transition_count',
        'history_length', 'history_unique_ratio', 'repeat_target',
        'text_last_cosine', 'text_max_cosine', 'sid_last_match_count',
        'sid_max_match_count',
    ]
    matrix = np.column_stack([
        np.log1p(data['target_train_count']),
        np.log1p(data['last_transition_count']),
        data['history_length'],
        data['history_unique_ratio'],
        data['repeat_target'].astype(np.float32),
        data['text_last_cosine'],
        data['text_max_cosine'],
        data['sid_last_match_count'],
        data['sid_max_match_count'],
    ])
    return names, matrix.astype(np.float32)


def score_geometry_features(data):
    names = [
        'drafter_ar_score_correlation', 'drafter_candidate_entropy',
        'ar_candidate_entropy', 'drafter_ar_top10_overlap',
        'drafter_fusion_top10_overlap', 'ar_fusion_top10_overlap',
    ]
    return names, np.column_stack([data[name] for name in names]).astype(np.float32)


def specialization_rows(data):
    drafter_hit = (data['drafter_rank'] > 0) & (data['drafter_rank'] <= 10)
    ar_hit = (data['ar_rank'] > 0) & (data['ar_rank'] <= 10)
    mask = drafter_hit ^ ar_hit
    label = ar_hit[mask].astype(np.int64)
    return mask, label


def ndcg10_from_rank(rank):
    rank = np.asarray(rank)
    result = np.zeros(rank.shape, dtype=np.float64)
    hit = (rank > 0) & (rank <= 10)
    result[hit] = 1.0 / np.log2(rank[hit] + 1.0)
    return result


def routing_experiment(name, val_features, test_features, val_data, test_data, seed):
    val_features_all = val_features
    test_features_all = test_features
    val_mask, val_label = specialization_rows(val_data)
    test_mask, test_label = specialization_rows(test_data)
    val_features = val_features[val_mask]
    test_features = test_features[test_mask]
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    candidates = (0.1, 1.0, 10.0)
    best = None
    for c_value in candidates:
        model = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(
                C=c_value, max_iter=3000, class_weight='balanced',
                random_state=seed, solver='liblinear',
            ),
        )
        score = float(cross_val_score(
            model, val_features, val_label, cv=folds, scoring='roc_auc'
        ).mean())
        if best is None or score > best[0]:
            best = (score, c_value, model)
    val_auc, c_value, model = best
    model.fit(val_features, val_label)
    probability = model.predict_proba(test_features)[:, 1]
    prediction = probability >= 0.5
    val_probability_all = model.predict_proba(
        np.asarray(val_features_all)
    )[:, 1]
    test_probability_all = model.predict_proba(
        np.asarray(test_features_all)
    )[:, 1]
    best_threshold = None
    best_validation_ndcg = float('-inf')
    for threshold in np.linspace(0.05, 0.95, 181):
        val_rank = np.where(
            val_probability_all >= threshold,
            val_data['ar_rank'], val_data['drafter_rank'],
        )
        score = float(ndcg10_from_rank(val_rank).mean())
        if score > best_validation_ndcg:
            best_validation_ndcg = score
            best_threshold = float(threshold)
    routed_rank = np.where(
        test_probability_all >= best_threshold,
        test_data['ar_rank'], test_data['drafter_rank'],
    )
    routed_hit = (routed_rank > 0) & (routed_rank <= 10)
    return {
        'feature_set': name,
        'validation_cv_auc': val_auc,
        'selected_C': c_value,
        'test_auc': float(roc_auc_score(test_label, probability)),
        'test_balanced_accuracy': float(
            balanced_accuracy_score(test_label, prediction)
        ),
        'route_threshold_selected_on_validation': best_threshold,
        'validation_routed_ndcg@10': best_validation_ndcg,
        'test_routed_recall@10': float(routed_hit.mean()),
        'test_routed_ndcg@10': float(ndcg10_from_rank(routed_rank).mean()),
        'validation_examples': int(len(val_label)),
        'test_examples': int(len(test_label)),
    }


class CoordinateProbe(nn.Module):
    def __init__(self, dimension, n_digit, codebook_size):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(dimension, codebook_size) for _ in range(n_digit)
        ])

    def forward(self, features):
        return torch.stack([head(features) for head in self.heads], dim=1)


def train_coordinate_probe(
    name, val_features, test_features, val_codes, test_codes,
    device, epochs, batch_size, seed,
):
    torch.manual_seed(seed)
    val_features = np.asarray(val_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    mean = val_features.mean(axis=0, keepdims=True)
    std = val_features.std(axis=0, keepdims=True)
    val_features = (val_features - mean) / np.maximum(std, 1e-5)
    test_features = (test_features - mean) / np.maximum(std, 1e-5)
    x_train = torch.from_numpy(val_features).to(device)
    y_train = torch.from_numpy(val_codes.astype(np.int64)).to(device)
    x_test = torch.from_numpy(test_features).to(device)
    y_test = torch.from_numpy(test_codes.astype(np.int64)).to(device)
    n_digit = y_train.shape[1]
    codebook_size = int(max(y_train.max(), y_test.max()).item()) + 1
    model = CoordinateProbe(
        x_train.shape[1], n_digit, codebook_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    last_loss = None
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_train), generator=generator)
        losses = []
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size].to(device)
            logits = model(x_train[rows])
            loss = torch.stack([
                F.cross_entropy(logits[:, digit], y_train[rows, digit])
                for digit in range(n_digit)
            ]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
    model.eval()
    correct_chunks = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            logits = model(x_test[start:start + batch_size])
            correct_chunks.append(
                logits.argmax(dim=-1).eq(y_test[start:start + batch_size]).cpu()
            )
    correct = torch.cat(correct_chunks).numpy()
    return {
        'feature_set': name,
        'parameters': int(sum(parameter.numel() for parameter in model.parameters())),
        'final_train_loss': last_loss,
        'coordinate_accuracy': [float(correct[:, digit].mean()) for digit in range(n_digit)],
        'exact_tuple_accuracy': float(correct.all(axis=1).mean()),
        'mean_coordinate_accuracy': float(correct.mean()),
    }


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
            elif isinstance(value, list):
                value = ', '.join(f'{entry:.4f}' for entry in value)
            values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def render_report(result):
    cka_rows = [
        {'comparison': name, 'linear_CKA': value}
        for name, value in result['linear_cka'].items()
    ]
    return '\n'.join([
        '# Representation complementarity report',
        '',
        '## Linear CKA on aligned test histories',
        '',
        markdown_table(cka_rows, ['comparison', 'linear_CKA']),
        '',
        '## Validation-to-test specialization routing',
        '',
        markdown_table(
            result['routing'],
            ['feature_set', 'validation_cv_auc', 'selected_C', 'test_auc',
             'test_balanced_accuracy', 'route_threshold_selected_on_validation',
             'test_routed_recall@10', 'test_routed_ndcg@10',
             'validation_examples', 'test_examples'],
        ),
        '',
        '## Frozen-representation linear coordinate probes',
        '',
        markdown_table(
            result['coordinate_probes'],
            ['feature_set', 'parameters', 'final_train_loss',
             'coordinate_accuracy', 'mean_coordinate_accuracy',
             'exact_tuple_accuracy'],
        ),
        '',
    ])


def main():
    args = parse_args()
    val = load_artifact(args.validation_artifact)
    test = load_artifact(args.test_artifact)
    device = torch.device(args.device)

    drafter_val = val['drafter_history_state'].astype(np.float32)
    drafter_test = test['drafter_history_state'].astype(np.float32)
    ar_val = val['ar_history_state'].astype(np.float32)
    ar_test = test['ar_history_state'].astype(np.float32)
    coordinate_val = val['drafter_coordinate_states'].astype(np.float32).reshape(len(drafter_val), -1)
    coordinate_test = test['drafter_coordinate_states'].astype(np.float32).reshape(len(drafter_test), -1)

    cka = {
        'drafter_history_vs_ar_history': centered_linear_cka(
            drafter_test, ar_test, args.cka_samples, args.seed
        ),
    }
    for digit in range(test['drafter_coordinate_states'].shape[1]):
        cka[f'drafter_coordinate_{digit}_vs_ar_history'] = centered_linear_cka(
            test['drafter_coordinate_states'][:, digit], ar_test,
            args.cka_samples, args.seed,
        )
    print('[representation] CKA complete', flush=True)

    _, external_val = external_features(val)
    _, external_test = external_features(test)
    _, score_val = score_geometry_features(val)
    _, score_test = score_geometry_features(test)
    routing_sets = {
        'external_attributes': (external_val, external_test),
        'score_geometry': (score_val, score_test),
        'drafter_history': (drafter_val, drafter_test),
        'ar_history': (ar_val, ar_test),
        'history_concat': (
            np.concatenate([drafter_val, ar_val], axis=1),
            np.concatenate([drafter_test, ar_test], axis=1),
        ),
        'history_plus_score': (
            np.concatenate([drafter_val, ar_val, score_val], axis=1),
            np.concatenate([drafter_test, ar_test, score_test], axis=1),
        ),
    }
    routing = [
        routing_experiment(name, left, right, val, test, args.seed)
        for name, (left, right) in routing_sets.items()
    ]
    print('[representation] specialization routing complete', flush=True)

    probe_sets = {
        'drafter_history': (drafter_val, drafter_test),
        'ar_history': (ar_val, ar_test),
        'history_concat': (
            np.concatenate([drafter_val, ar_val], axis=1),
            np.concatenate([drafter_test, ar_test], axis=1),
        ),
        'drafter_coordinate_states': (coordinate_val, coordinate_test),
    }
    probes = []
    for name, (left, right) in probe_sets.items():
        probes.append(train_coordinate_probe(
            name, left, right, val['target_codes'], test['target_codes'],
            device, args.probe_epochs, args.probe_batch_size, args.seed,
        ))
        print(f'[representation] probe complete: {name}', flush=True)

    result = {
        'validation_artifact': str(Path(args.validation_artifact).resolve()),
        'test_artifact': str(Path(args.test_artifact).resolve()),
        'linear_cka': cka,
        'routing': routing,
        'coordinate_probes': probes,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'representation_analysis.json', 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    (output_dir / 'REPRESENTATION_REPORT.md').write_text(
        render_report(result), encoding='utf-8'
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
