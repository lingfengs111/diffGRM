#!/usr/bin/env python
"""One discrete E-step for decoder-aware, collision-free OPQ assignments.

Existing unique catalog SIDs are permuted inside small semantic-neighbourhood
groups.  The assignment cost combines AR path NLL on training histories with
text-embedding distance.  Since the operation is a permutation, uniqueness,
code utilization and catalog size are preserved exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from accelerate import Accelerator
import faiss
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2014')
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--semantic-weights', default='4,1,0.25')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def semantic_groups(embeddings: np.ndarray, group_size: int):
    """Deterministic greedy nearest-neighbour partition."""
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    available = np.ones(len(embeddings), dtype=bool)
    groups = []
    for seed in range(len(embeddings)):
        if not available[seed]:
            continue
        search_k = min(len(embeddings), max(group_size * 8, group_size))
        _, neighbours = index.search(embeddings[seed:seed + 1], search_k)
        group = [int(idx) for idx in neighbours[0] if available[int(idx)]][:group_size]
        if len(group) < group_size:
            fill = np.flatnonzero(available)
            group.extend(int(idx) for idx in fill if int(idx) not in group)
            group = group[:group_size]
        available[np.asarray(group)] = False
        groups.append(np.asarray(group, dtype=np.int64))
    return groups


def normalize_rows(cost: np.ndarray):
    shifted = cost - cost.min(axis=1, keepdims=True)
    scale = shifted.std(axis=1, keepdims=True)
    return shifted / np.maximum(scale, 1e-6)


def main():
    args = parse_args()
    accelerator = Accelerator()
    config = get_config('AR_GRM', args.dataset, args.config, {})
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator

    dataset = get_dataset(args.dataset)(config)
    splits = dataset.split()
    tokenizer = AR_GRMTokenizer(config, dataset)
    tokenized = tokenizer.tokenize(splits)
    codes = catalog_codes(tokenizer, config['codebook_size'])
    if len(np.unique(codes, axis=0)) != len(codes):
        raise ValueError('base catalog must be collision free')

    emb_path = (
        Path(dataset.cache_dir) / 'processed' /
        f'{Path(config["sent_emb_model"]).name}_pca{config["sent_emb_pca"]}.sent_emb'
    )
    embeddings = np.fromfile(emb_path, dtype=np.float32).reshape(
        -1, int(config['sent_emb_pca'])
    )
    if len(embeddings) != len(codes):
        raise ValueError(f'embedding/catalog mismatch: {embeddings.shape} vs {codes.shape}')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    groups = semantic_groups(embeddings, args.group_size)
    item_to_group = np.empty(len(codes), dtype=np.int64)
    item_to_local = np.empty(len(codes), dtype=np.int64)
    for group_idx, members in enumerate(groups):
        item_to_group[members] = group_idx
        item_to_local[members] = np.arange(len(members))
    max_group = max(map(len, groups))

    model = AR_GRM(config, dataset, tokenizer).to(config['device'])
    model.load_state_dict(torch.load(args.checkpoint, map_location=config['device']))
    model.eval()

    code_to_item = {tuple(row.tolist()): idx for idx, row in enumerate(codes)}
    score_sums = [np.zeros((len(g), len(g)), dtype=np.float64) for g in groups]
    target_counts = np.zeros(len(codes), dtype=np.int64)
    loader = DataLoader(
        tokenized['train'], batch_size=args.batch_size, shuffle=False,
        collate_fn=tokenizer.collate_fn['train'],
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc='Scoring decoder-aware SID candidates'):
            labels = batch['decoder_labels'].cpu().numpy()
            target_items = np.asarray(
                [code_to_item[tuple(row.tolist())] for row in labels], dtype=np.int64
            )
            candidate_rows = np.empty(
                (len(target_items), max_group, codes.shape[1]), dtype=np.int64
            )
            valid_widths = []
            for row_idx, item_idx in enumerate(target_items):
                members = groups[item_to_group[item_idx]]
                width = len(members)
                candidate_rows[row_idx, :width] = codes[members]
                candidate_rows[row_idx, width:] = codes[item_idx]
                valid_widths.append(width)
            scores = model.score_candidate_paths(
                batch, torch.from_numpy(candidate_rows), chunk_size=max_group
            ).detach().cpu().numpy()
            for row_idx, item_idx in enumerate(target_items):
                group_idx = int(item_to_group[item_idx])
                local_idx = int(item_to_local[item_idx])
                width = valid_widths[row_idx]
                score_sums[group_idx][local_idx, :width] += scores[row_idx, :width]
                target_counts[item_idx] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'checkpoint': args.checkpoint,
        'group_size': args.group_size,
        'n_groups': len(groups),
        'n_observed_target_items': int((target_counts > 0).sum()),
        'n_train_examples': int(target_counts.sum()),
        'variants': {},
    }
    semantic_weights = [float(value) for value in args.semantic_weights.split(',')]
    for semantic_weight in semantic_weights:
        reassigned = codes.copy()
        moved = 0
        semantic_cosines = []
        decoder_deltas = []
        for group_idx, members in enumerate(groups):
            group_scores = score_sums[group_idx]
            counts = target_counts[members]
            mean_scores = np.divide(
                group_scores, counts[:, None],
                out=np.zeros_like(group_scores), where=counts[:, None] > 0,
            )
            decoder_cost = -mean_scores
            observed = counts > 0
            if observed.any():
                decoder_cost[observed] = normalize_rows(decoder_cost[observed])
            decoder_cost[~observed] = 0.0

            cosine = embeddings[members] @ embeddings[members].T
            semantic_cost = np.maximum(0.0, 1.0 - cosine)
            positive = semantic_cost[semantic_cost > 1e-8]
            if len(positive):
                semantic_cost = semantic_cost / max(float(np.median(positive)), 1e-6)
            total_cost = decoder_cost + semantic_weight * semantic_cost
            rows, cols = linear_sum_assignment(total_cost)
            if not np.array_equal(rows, np.arange(len(members))):
                raise RuntimeError('unexpected Hungarian row ordering')
            reassigned[members] = codes[members[cols]]
            moved += int((cols != np.arange(len(members))).sum())
            semantic_cosines.extend(cosine[np.arange(len(members)), cols].tolist())
            if observed.any():
                original = -mean_scores[np.arange(len(members)), np.arange(len(members))]
                assigned = -mean_scores[np.arange(len(members)), cols]
                decoder_deltas.extend((assigned[observed] - original[observed]).tolist())

        if len(np.unique(reassigned, axis=0)) != len(reassigned):
            raise RuntimeError('assignment unexpectedly introduced collisions')
        tag = f'decaware_sem{semantic_weight:g}'.replace('.', 'p')
        output_path = output_dir / f'{tag}.sem_ids.json'
        payload = {
            tokenizer.id2item[item_idx + 1]: reassigned[item_idx].tolist()
            for item_idx in range(len(reassigned))
        }
        output_path.write_text(json.dumps(payload))
        variant_report = {
            'path': str(output_path.resolve()),
            'semantic_weight': semantic_weight,
            'moved_items': moved,
            'moved_ratio': moved / len(codes),
            'mean_assigned_text_cosine': float(np.mean(semantic_cosines)),
            'mean_train_path_nll_delta': float(np.mean(decoder_deltas)),
            'unique_sid_ratio': 1.0,
        }
        report['variants'][tag] = variant_report
        print(json.dumps({tag: variant_report}, indent=2))

    report_path = output_dir / 'assignment_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    print(f'wrote {report_path}')


if __name__ == '__main__':
    main()
