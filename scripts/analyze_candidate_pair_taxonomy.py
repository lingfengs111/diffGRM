#!/usr/bin/env python
"""Decompose drafter/AR complementarity by target-negative SID geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from accelerate import Accelerator
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.utils import get_dataset
from scripts.train_parallel_opq_drafter import make_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def resolve(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def rebuild_catalog(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    common = [str(resolve(args['common_config']))]
    if args.get('sid_config'):
        common.append(str(resolve(args['sid_config'])))
    config = make_config(
        'AR_GRM', args['dataset'],
        common + [str(resolve(args['ar_config']))],
        Accelerator(), {},
    )
    dataset = get_dataset(args['dataset'])(config)
    tokenizer = AR_GRMTokenizer(config, dataset)
    return catalog_codes(tokenizer, config['codebook_size']).astype(np.int16)


def normalize(value):
    value = np.asarray(value, dtype=np.float32)
    return (value - value.mean(axis=1, keepdims=True)) / np.maximum(
        value.std(axis=1, keepdims=True), 1e-6
    )


def longest_common_prefix(candidate, target):
    equal = candidate == target[:, None, :]
    return np.cumprod(equal.astype(np.int8), axis=2).sum(axis=2)


def group_pair_statistics(label, group, valid, margins):
    rows = []
    for value in sorted(np.unique(group[valid]).tolist()):
        mask = valid & (group == value)
        count = int(mask.sum())
        drafter_correct = margins['drafter'][mask] > 0
        ar_correct = margins['ar'][mask] > 0
        fusion_correct = margins['fusion'][mask] > 0
        rows.append({
            'grouping': label,
            'group': int(value),
            'pair_count': count,
            'fraction': float(count / valid.sum()),
            'drafter_pair_accuracy': float(drafter_correct.mean()),
            'ar_pair_accuracy': float(ar_correct.mean()),
            'fusion_pair_accuracy': float(fusion_correct.mean()),
            'ar_rescues_drafter_pair': float((~drafter_correct & ar_correct).mean()),
            'drafter_rescues_ar_pair': float((drafter_correct & ~ar_correct).mean()),
            'mean_drafter_margin': float(margins['drafter'][mask].mean()),
            'mean_ar_margin': float(margins['ar'][mask].mean()),
            'mean_fusion_margin': float(margins['fusion'][mask].mean()),
        })
    return rows


def hard_negative_distribution(scores, codes, target, present, label):
    wrong = np.any(codes != target[:, None, :], axis=2)
    best_index = np.where(wrong, scores, -np.inf).argmax(axis=1)
    selected = codes[np.arange(len(codes)), best_index]
    matches = selected == target
    lcp = np.cumprod(matches.astype(np.int8), axis=1).sum(axis=1)
    count = matches.sum(axis=1)
    rows = []
    for grouping, values in [('prefix_length', lcp), ('matching_coordinates', count)]:
        for value in sorted(np.unique(values[present]).tolist()):
            mask = present & (values == value)
            rows.append({
                'scorer': label,
                'grouping': grouping,
                'group': int(value),
                'count': int(mask.sum()),
                'fraction': float(mask.sum() / present.sum()),
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


def main():
    args = parse_args()
    artifact = resolve(args.artifact_dir)
    checkpoint = resolve(args.drafter_checkpoint)
    with np.load(artifact / 'examples.npz') as handle:
        data = {key: handle[key] for key in handle.files}
    catalog = rebuild_catalog(checkpoint)
    candidate_ids = data['proposal_item_ids']
    candidate_codes = catalog[candidate_ids - 1]
    target_codes = data['target_codes'].astype(np.int16)
    exact = np.all(candidate_codes == target_codes[:, None, :], axis=2)
    present = exact.any(axis=1)
    target_position = exact.argmax(axis=1)

    scores = {
        'drafter': normalize(data['drafter_candidate_scores']),
        'ar': normalize(data['ar_candidate_scores']),
        'fusion': normalize(data['fusion_candidate_scores']),
    }
    margins = {}
    for name, score in scores.items():
        target_score = score[np.arange(len(score)), target_position]
        margins[name] = target_score[:, None] - score

    wrong = ~exact
    valid = wrong & present[:, None]
    lcp = longest_common_prefix(candidate_codes, target_codes)
    matches = (candidate_codes == target_codes[:, None, :]).sum(axis=2)
    pair_rows = (
        group_pair_statistics('prefix_length', lcp, valid, margins)
        + group_pair_statistics('matching_coordinates', matches, valid, margins)
    )
    hard_rows = []
    for name, score in scores.items():
        hard_rows.extend(hard_negative_distribution(
            score, candidate_codes, target_codes, present, name
        ))

    result = {
        'n_examples': int(len(target_codes)),
        'candidate_present_examples': int(present.sum()),
        'candidate_recall': float(present.mean()),
        'pair_statistics': pair_rows,
        'hard_negative_distribution': hard_rows,
        'note': (
            'prefix_length follows the fixed verifier order c0->c1->c2->c3; '
            'it is not assumed to be a semantic hierarchy for OPQ.'
        ),
    }
    output_dir = resolve(args.output_dir) if args.output_dir else artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'candidate_pair_taxonomy.json', 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    report = '\n'.join([
        '# Candidate-pair SID taxonomy', '', result['note'], '',
        '## Pairwise target-versus-negative decisions', '',
        markdown_table(pair_rows, [
            'grouping', 'group', 'pair_count', 'fraction',
            'drafter_pair_accuracy', 'ar_pair_accuracy', 'fusion_pair_accuracy',
            'ar_rescues_drafter_pair', 'drafter_rescues_ar_pair',
            'mean_drafter_margin', 'mean_ar_margin', 'mean_fusion_margin',
        ]), '',
        '## Highest-scoring wrong candidate geometry', '',
        markdown_table(hard_rows, [
            'scorer', 'grouping', 'group', 'count', 'fraction'
        ]), '',
    ])
    (output_dir / 'CANDIDATE_PAIR_TAXONOMY.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
