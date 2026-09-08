#!/usr/bin/env python
"""Diagnose OPQ coordinates under every visible-coordinate subset.

The diagnostic deliberately separates representation geometry from decoder
quality.  It evaluates all 15 non-full decoder states for a four-coordinate
SID (including the all-masked state), all 15 non-empty catalog subsets, and all
24 teacher-forced reveal orders.  Known coordinates always come from the true
target, so every result involving a non-empty known set is an oracle analysis,
not a deployable recommendation metric.
"""

import argparse
from collections import Counter
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import (
    catalog_codes,
    catalog_subset_diagnostics,
    coordinate_subset_masks,
    stable_target_ranks,
    subset_digits,
    subset_key,
)
from genrec.utils import get_config, get_dataset, get_model, get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DIFF_GRM')
    parser.add_argument('--dataset', default='AmazonReviews2014CleanGR')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', action='append', default=None)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--ranking-ks', default='1,5,10,20')
    parser.add_argument('--embedding-path', default=None)
    parser.add_argument('--knn-sample-size', type=int, default=4096)
    parser.add_argument('--knn-ks', default='1,5,10,20')
    parser.add_argument('--knn-batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def parse_ints(value):
    values = sorted({int(part) for part in str(value).split(',') if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError(f'expected positive comma-separated integers, got {value!r}')
    return values


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {'count': 0}
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'p99': float(np.percentile(values, 99)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def neighbor_agreement(codes, query_rows, neighbor_rows, ks):
    query = codes[query_rows]
    neighbors = codes[neighbor_rows]
    n_digit = codes.shape[1]
    output = {}
    for k in ks:
        current = neighbors[:, :min(k, neighbors.shape[1])]
        coordinate = current == query[:, None, :]
        item = {'coordinate_match_rate': {}}
        for digit in range(n_digit):
            item['coordinate_match_rate'][str(digit)] = float(coordinate[:, :, digit].mean())
        item['subset_match_rate'] = {}
        for mask in coordinate_subset_masks(n_digit, include_empty=False, include_full=True):
            digits = subset_digits(mask, n_digit)
            item['subset_match_rate'][subset_key(mask, n_digit)] = float(
                coordinate[:, :, digits].all(axis=-1).mean()
            )
        output[str(k)] = item
    return output


@torch.inference_mode()
def embedding_knn_diagnostics(path, dim, codes, sample_size, ks, batch_size, seed, device):
    embeddings = np.fromfile(path, dtype=np.float32)
    if embeddings.size % int(dim):
        raise ValueError(f'embedding file size {embeddings.size} is not divisible by dim={dim}')
    embeddings = embeddings.reshape(-1, int(dim))
    if embeddings.shape[0] != codes.shape[0]:
        raise ValueError(f'embedding/catalog rows differ: {embeddings.shape[0]} != {codes.shape[0]}')
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    rng = np.random.default_rng(seed)
    n_items = len(codes)
    n_query = min(int(sample_size), n_items)
    query_rows = np.sort(rng.choice(n_items, size=n_query, replace=False))
    max_k = min(max(ks), n_items - 1)
    table = torch.from_numpy(embeddings).to(device)
    batches = []
    for start in tqdm(range(0, n_query, batch_size), desc='raw embedding kNN'):
        rows_np = query_rows[start:start + batch_size]
        rows = torch.tensor(rows_np, device=device, dtype=torch.long)
        scores = table.index_select(0, rows) @ table.t()
        scores[torch.arange(len(rows), device=device), rows] = float('-inf')
        batches.append(scores.topk(max_k, dim=1).indices.cpu().numpy())
    neighbors = np.concatenate(batches, axis=0)

    random_neighbors = np.empty_like(neighbors)
    for row_idx, query_row in enumerate(query_rows):
        draw = rng.choice(n_items - 1, size=max_k, replace=False)
        random_neighbors[row_idx] = draw + (draw >= query_row)
    return {
        'path': str(path),
        'dim': int(dim),
        'sample_size': int(n_query),
        'ks': ks,
        'raw_knn': neighbor_agreement(codes, query_rows, neighbors, ks),
        'random_neighbors': neighbor_agreement(codes, query_rows, random_neighbors, ks),
    }


def build_target_rows(labels, code_to_row):
    rows = []
    for label in labels.detach().cpu().tolist():
        key = tuple(int(value) for value in label)
        if key not in code_to_row:
            raise KeyError(f'target SID is absent from the catalog: {key}')
        rows.append(code_to_row[key])
    return torch.tensor(rows, dtype=torch.long, device=labels.device)


def main():
    args = parse_args()
    ranking_ks = parse_ints(args.ranking_ks)
    knn_ks = parse_ints(args.knn_ks)
    accelerator = Accelerator()
    config = get_config(
        args.model,
        args.dataset,
        args.config,
        {
            'eval_batch_size': args.batch_size,
            'force_regenerate_opq': False,
            'force_regenerate_codes': False,
        },
    )
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator
    config['current_split'] = args.split
    device = torch.device(config['device'])

    dataset = get_dataset(args.dataset)(config)
    splits = dataset.split()
    tokenizer = get_tokenizer(args.model)(config, dataset)
    tokenized = tokenizer.tokenize(splits)
    model = get_model(args.model)(config, dataset, tokenizer).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    codes_np = catalog_codes(tokenizer, config['codebook_size'])
    if len(np.unique(codes_np, axis=0)) != len(codes_np):
        raise ValueError('subset oracle requires a collision-free item catalog')
    codes = torch.tensor(codes_np, dtype=torch.long, device=device)
    code_to_row = {tuple(int(value) for value in row): idx for idx, row in enumerate(codes_np)}
    n_digit = int(codes.shape[1])
    full_mask = (1 << n_digit) - 1
    decoder_states = coordinate_subset_masks(
        n_digit, include_empty=True, include_full=False
    )
    state_to_index = {mask: idx for idx, mask in enumerate(decoder_states)}
    visible = torch.tensor(
        [[bool(mask & (1 << digit)) for digit in range(n_digit)] for mask in decoder_states],
        dtype=torch.bool,
        device=device,
    )
    missing = ~visible
    reveal_orders = list(permutations(range(n_digit)))

    eval_data = tokenized[args.split]
    if args.max_examples is not None:
        eval_data = eval_data.select(range(min(int(args.max_examples), len(eval_data))))
    loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn[args.split],
    )

    n_state = len(decoder_states)
    conditional_count = np.zeros((n_state, n_digit), dtype=np.int64)
    conditional_acc = np.zeros((n_state, n_digit), dtype=np.float64)
    conditional_nll = np.zeros((n_state, n_digit), dtype=np.float64)
    conditional_mrr = np.zeros((n_state, n_digit), dtype=np.float64)
    conditional_entropy = np.zeros((n_state, n_digit), dtype=np.float64)
    ranking_count = np.zeros(n_state, dtype=np.int64)
    ranking_mrr = np.zeros(n_state, dtype=np.float64)
    ranking_hits = {k: np.zeros(n_state, dtype=np.int64) for k in ranking_ks}
    target_group_sizes = [[] for _ in decoder_states]
    order_nll = np.zeros(len(reveal_orders), dtype=np.float64)
    order_exact = np.zeros(len(reveal_orders), dtype=np.int64)
    order_count = 0
    best_order_nll = 0.0
    worst_order_nll = 0.0
    order_gap = 0.0
    best_order_frequency = Counter()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f'OPQ subset oracle {args.split}'):
            labels = batch['labels'].to(device).long()
            batch_size = labels.shape[0]
            target_rows = build_target_rows(labels, code_to_row)
            encoder_hidden = model(batch, return_loss=False).hidden_states

            decoder_inputs = torch.where(
                visible[None, :, :],
                labels[:, None, :],
                torch.zeros(batch_size, n_state, n_digit, dtype=torch.long, device=device),
            )
            mask_positions = missing[None, :, :].expand(batch_size, -1, -1)
            flat_encoder = encoder_hidden[:, None, :, :].expand(
                -1, n_state, -1, -1
            ).reshape(batch_size * n_state, encoder_hidden.shape[1], encoder_hidden.shape[2])
            outputs = model.forward_decoder_only(
                {
                    'decoder_input_ids': decoder_inputs.reshape(-1, n_digit),
                    'encoder_hidden': flat_encoder,
                    'mask_positions': mask_positions.reshape(-1, n_digit).float(),
                },
                return_loss=False,
                digit=None,
                use_cache=False,
            )
            logits = outputs.logits.reshape(
                batch_size, n_state, n_digit, config['codebook_size']
            ).float()
            log_probs = F.log_softmax(logits, dim=-1)
            expanded_labels = labels[:, None, :, None].expand(-1, n_state, -1, -1)
            target_logp = log_probs.gather(-1, expanded_labels).squeeze(-1)
            target_logits = logits.gather(-1, expanded_labels).squeeze(-1)
            ranks = logits.gt(target_logits.unsqueeze(-1)).sum(dim=-1) + 1
            predictions = logits.argmax(dim=-1)
            probs = log_probs.exp()
            norm_entropy = -(probs * log_probs).sum(dim=-1) / np.log(config['codebook_size'])

            for state_idx in range(n_state):
                for digit in range(n_digit):
                    if not bool(missing[state_idx, digit]):
                        continue
                    conditional_count[state_idx, digit] += batch_size
                    conditional_acc[state_idx, digit] += float(
                        predictions[:, state_idx, digit].eq(labels[:, digit]).sum()
                    )
                    conditional_nll[state_idx, digit] += float(-target_logp[:, state_idx, digit].sum())
                    conditional_mrr[state_idx, digit] += float(
                        ranks[:, state_idx, digit].float().reciprocal().sum()
                    )
                    conditional_entropy[state_idx, digit] += float(
                        norm_entropy[:, state_idx, digit].sum()
                    )

            # Score every catalog item with the simultaneous conditional logits
            # of the still-masked coordinates, then restrict to items matching
            # the oracle-visible target coordinates.
            catalog_scores = torch.zeros(
                batch_size, n_state, codes.shape[0], dtype=torch.float, device=device
            )
            valid = torch.ones(
                batch_size, n_state, codes.shape[0], dtype=torch.bool, device=device
            )
            for digit in range(n_digit):
                digit_scores = log_probs[:, :, digit, :].index_select(2, codes[:, digit])
                catalog_scores += digit_scores * missing[None, :, digit, None]
                coordinate_match = labels[:, None, digit, None].eq(
                    codes[None, None, :, digit]
                )
                valid &= (~visible[None, :, digit, None]) | coordinate_match
            catalog_scores.masked_fill_(~valid, float('-inf'))
            for state_idx in range(n_state):
                current_ranks = stable_target_ranks(
                    catalog_scores[:, state_idx], target_rows
                )
                ranking_count[state_idx] += batch_size
                ranking_mrr[state_idx] += float(current_ranks.float().reciprocal().sum())
                for k in ranking_ks:
                    ranking_hits[k][state_idx] += int(current_ranks.le(k).sum())
                target_group_sizes[state_idx].append(
                    valid[:, state_idx].sum(dim=1).detach().cpu().numpy()
                )

            batch_order_nll = []
            batch_order_exact = []
            for order_idx, order in enumerate(reveal_orders):
                state = 0
                current_nll = torch.zeros(batch_size, device=device)
                current_exact = torch.ones(batch_size, dtype=torch.bool, device=device)
                for digit in order:
                    state_idx = state_to_index[state]
                    current_nll -= target_logp[:, state_idx, digit]
                    current_exact &= predictions[:, state_idx, digit].eq(labels[:, digit])
                    state |= 1 << digit
                order_nll[order_idx] += float(current_nll.sum())
                order_exact[order_idx] += int(current_exact.sum())
                batch_order_nll.append(current_nll)
                batch_order_exact.append(current_exact)
            stacked_order_nll = torch.stack(batch_order_nll, dim=1)
            best_values, best_indices = stacked_order_nll.min(dim=1)
            worst_values = stacked_order_nll.max(dim=1).values
            best_order_nll += float(best_values.sum())
            worst_order_nll += float(worst_values.sum())
            order_gap += float((worst_values - best_values).sum())
            best_order_frequency.update(best_indices.detach().cpu().tolist())
            order_count += batch_size

    subset_results = {}
    for state_idx, mask in enumerate(decoder_states):
        known = subset_digits(mask, n_digit)
        unknown = [digit for digit in range(n_digit) if digit not in known]
        coordinate_results = {}
        for digit in unknown:
            count = max(1, int(conditional_count[state_idx, digit]))
            coordinate_results[str(digit)] = {
                'accuracy': float(conditional_acc[state_idx, digit] / count),
                'nll': float(conditional_nll[state_idx, digit] / count),
                'mrr': float(conditional_mrr[state_idx, digit] / count),
                'normalized_entropy': float(conditional_entropy[state_idx, digit] / count),
            }
        count = max(1, int(ranking_count[state_idx]))
        subset_results[subset_key(mask, n_digit)] = {
            'mask': int(mask),
            'known_digits': known,
            'unknown_digits': unknown,
            'remaining_coordinate_metrics': coordinate_results,
            'catalog_rank': {
                'mrr': float(ranking_mrr[state_idx] / count),
                **{
                    f'recall@{k}': float(ranking_hits[k][state_idx] / count)
                    for k in ranking_ks
                },
            },
            'oracle_candidate_group_size': describe(
                np.concatenate(target_group_sizes[state_idx])
            ),
        }

    order_results = []
    for order_idx, order in enumerate(reveal_orders):
        order_results.append({
            'order': list(order),
            'mean_target_path_nll': float(order_nll[order_idx] / max(1, order_count)),
            'teacher_forced_full_argmax_accuracy': float(
                order_exact[order_idx] / max(1, order_count)
            ),
            'best_nll_frequency': float(
                best_order_frequency[order_idx] / max(1, order_count)
            ),
        })
    order_results.sort(key=lambda row: row['mean_target_path_nll'])

    summary = {
        'protocol': {
            'dataset': args.dataset,
            'split': args.split,
            'num_examples': int(len(eval_data)),
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': sha256(args.checkpoint),
            'n_digit': n_digit,
            'codebook_size': int(config['codebook_size']),
            'catalog_items': int(codes.shape[0]),
            'catalog_unique_sids': int(len(np.unique(codes_np, axis=0))),
            'sid_quantizer': config.get('sid_quantizer'),
            'sid_collision_strategy': config.get('sid_collision_strategy'),
            'ranking_score_definition': (
                'sum of simultaneous conditional log-probabilities over unknown coordinates, '
                'restricted to catalog items matching oracle-visible target coordinates'
            ),
            'history_postfilter': False,
        },
        'representation': catalog_subset_diagnostics(codes_np),
        'decoder_known_subset_oracle': subset_results,
        'teacher_forced_reveal_orders': order_results,
        'order_sensitivity': {
            'per_example_oracle_best_mean_path_nll': float(best_order_nll / max(1, order_count)),
            'per_example_worst_mean_path_nll': float(worst_order_nll / max(1, order_count)),
            'mean_worst_minus_best_path_nll': float(order_gap / max(1, order_count)),
        },
        'coordinate_summary': {},
    }
    for digit in range(n_digit):
        empty = subset_results['none']['remaining_coordinate_metrics'][str(digit)]
        leave_one_out_mask = full_mask ^ (1 << digit)
        leave_one_out = subset_results[subset_key(leave_one_out_mask, n_digit)][
            'remaining_coordinate_metrics'
        ][str(digit)]
        summary['coordinate_summary'][str(digit)] = {
            'all_masked': empty,
            'leave_one_out': leave_one_out,
        }

    if args.embedding_path:
        summary['embedding_knn'] = embedding_knn_diagnostics(
            path=Path(args.embedding_path),
            dim=int(config['sent_emb_pca'] or config['sent_emb_dim']),
            codes=codes_np,
            sample_size=args.knn_sample_size,
            ks=knn_ks,
            batch_size=args.knn_batch_size,
            seed=args.seed,
            device=device,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(output), **summary}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
