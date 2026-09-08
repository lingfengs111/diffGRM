#!/usr/bin/env python
"""Evaluate diffusion SID proposals reranked by an autoregressive verifier."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2014')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--sid-config', default=None)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--proposal-k', type=int, default=128)
    parser.add_argument(
        '--proposal-mode',
        choices=('random', 'catalog_uncertainty'),
        default='random',
    )
    parser.add_argument(
        '--decode-orders', default=None,
        help='Semicolon-separated permutations, e.g. 0,1,2,3;2,3,0,1. '
             'Candidates from all orders are unioned before AR verification.'
    )
    parser.add_argument('--output-k', type=int, default=10)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument(
        '--metric-ks',
        default='5,10',
        help='Comma-separated ranking cutoffs. output-k is always included.',
    )
    parser.add_argument(
        '--fusion-alphas',
        default='',
        help=(
            'Comma-separated AR weights for validation-selected score fusion. '
            'Diffusion weight is 1-alpha; scores are normalized per user.'
        ),
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def _config(model, dataset, files, accelerator, overrides=None):
    config = get_config(model, dataset, files, overrides or {})
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator
    return config


def _ranking_metrics(candidates, labels, k):
    matches = candidates.eq(labels[:, None, :]).all(dim=-1)
    cutoff = min(k, candidates.shape[1])
    has_hit = matches[:, :cutoff].any(dim=1)
    positions = torch.arange(candidates.shape[1], device=candidates.device)[None, :]
    first = torch.where(matches, positions, candidates.shape[1]).min(dim=1).values
    recall = has_hit.float()
    ndcg = torch.where(
        has_hit,
        torch.log2(first.float() + 2.0).reciprocal(),
        torch.zeros_like(first, dtype=torch.float),
    )
    return recall, ndcg


def _parse_orders(text, n_digit):
    if not text:
        return [None]
    orders = []
    for raw_order in text.split(';'):
        order = [int(value) for value in raw_order.split(',')]
        if sorted(order) != list(range(n_digit)):
            raise ValueError(f"invalid decode order: {order}")
        orders.append(order)
    return orders


def _parse_metric_ks(text, output_k):
    values = {int(output_k)}
    if text:
        values.update(int(value) for value in text.split(','))
    if any(value <= 0 for value in values):
        raise ValueError(f'metric cutoffs must be positive, got {sorted(values)}')
    return sorted(values)


def _parse_fusion_alphas(text):
    if not text:
        return []
    values = sorted({float(value) for value in text.split(',')})
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f'fusion alphas must lie in [0, 1], got {values}')
    return values


def _alpha_tag(alpha):
    return f'{alpha:g}'.replace('.', 'p')


def _normalize_candidate_scores(scores, valid):
    """Per-user z-normalization over legal proposal candidates only."""
    clean = scores.masked_fill(~valid, 0.0)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mean = clean.sum(dim=1, keepdim=True) / count
    centered = (scores - mean).masked_fill(~valid, 0.0)
    variance = centered.square().sum(dim=1, keepdim=True) / count
    normalized = centered / variance.sqrt().clamp_min(1e-6)
    return normalized.masked_fill(~valid, float('-inf'))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_candidates(candidate_sets, score_sets):
    """Stable union and log-sum-exp path scores across reveal orders.

    Generation pads an order with copies of its last legal candidate when it
    returns fewer than ``proposal_k`` unique paths.  Such copies are padding,
    not independent reveal-order evidence: deduplicate with max *within* each
    order, then log-sum-exp a candidate at most once per distinct order.
    """
    combined = torch.cat(candidate_sets, dim=1)
    combined_scores = torch.cat(score_sets, dim=1)
    per_user = []
    per_user_scores = []
    candidate_rows = [rows.detach().cpu().tolist() for rows in candidate_sets]
    candidate_scores = [scores.detach().cpu().tolist() for scores in score_sets]
    for user_idx in range(combined.shape[0]):
        across_orders = {}
        for order_rows, order_scores in zip(candidate_rows, candidate_scores):
            within_order = {}
            for row, score in zip(order_rows[user_idx], order_scores[user_idx]):
                key = tuple(int(value) for value in row)
                within_order[key] = max(within_order.get(key, float('-inf')), float(score))
            for key, score in within_order.items():
                if key in across_orders:
                    across_orders[key] = float(np.logaddexp(across_orders[key], score))
                else:
                    across_orders[key] = score
        per_user.append(list(across_orders))
        per_user_scores.append(list(across_orders.values()))
    max_candidates = max(len(rows) for rows in per_user)
    output = combined.new_zeros(combined.shape[0], max_candidates, combined.shape[-1])
    valid = torch.zeros(combined.shape[0], max_candidates, dtype=torch.bool, device=combined.device)
    aggregate_scores = combined_scores.new_full(
        (combined.shape[0], max_candidates), float('-inf')
    )
    for user_idx, (rows, scores) in enumerate(zip(per_user, per_user_scores)):
        stacked = torch.tensor(rows, dtype=combined.dtype, device=combined.device)
        output[user_idx, :len(rows)] = stacked
        valid[user_idx, :len(rows)] = True
        aggregate_scores[user_idx, :len(rows)] = torch.tensor(
            scores, dtype=combined_scores.dtype, device=combined.device
        )
    return output, valid, aggregate_scores


def main():
    args = parse_args()
    accelerator = Accelerator()
    common_files = [args.common_config]
    if args.sid_config:
        common_files.append(args.sid_config)
    ar_config = _config(
        'AR_GRM', args.dataset, common_files + [args.ar_config], accelerator,
        {'eval_batch_size': args.batch_size},
    )
    diffusion_config = _config(
        'DIFF_GRM', args.dataset, common_files + [args.diffusion_config], accelerator,
        {
            'eval_batch_size': args.batch_size,
            'beam_search_modes': ['random'],
            'random_beam': {
                'beam_act': args.proposal_k,
                'beam_max': args.proposal_k,
                'seed': 42,
            },
            'vectorized_beam_search': {
                'top_k_final': args.proposal_k,
                'neg_inf_fp32': -1e9,
                'neg_inf_fp16': -65504.0,
                'dedup_strategy': 'simple',
                'val': {'beam_act': args.proposal_k, 'beam_max': args.proposal_k},
                'test': {'beam_act': args.proposal_k, 'beam_max': args.proposal_k},
                'beam_act': args.proposal_k,
                'beam_max': args.proposal_k,
            },
        },
    )
    device = torch.device(diffusion_config['device'])

    dataset = get_dataset(args.dataset)(ar_config)
    splits = dataset.split()
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    tokenized = tokenizer.tokenize(splits)
    codes = catalog_codes(tokenizer, ar_config['codebook_size'])
    digest = hashlib.sha256(codes.tobytes()).hexdigest()
    n_catalog_items = int(codes.shape[0])
    n_unique_sids = int(np.unique(codes, axis=0).shape[0])
    ar_checkpoint_sha256 = _sha256(args.ar_checkpoint)
    diffusion_checkpoint_sha256 = _sha256(args.diffusion_checkpoint)

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    diffusion_model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    diffusion_model.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))
    ar_model.eval()
    diffusion_model.eval()
    diffusion_model.config['current_split'] = args.split

    eval_data = tokenized[args.split]
    if args.max_examples is not None:
        eval_data = eval_data.select(range(min(args.max_examples, len(eval_data))))
    loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn[args.split],
    )

    decode_orders = _parse_orders(args.decode_orders, ar_config['n_digit'])
    metric_ks = _parse_metric_ks(args.metric_ks, args.output_k)
    fusion_alphas = _parse_fusion_alphas(args.fusion_alphas)
    aggregates = {
        'proposal_recall': [],
        'unique_candidates': [],
    }
    for k in metric_ks:
        for source in ('diffusion', 'diffusion_union', 'verified'):
            aggregates[f'{source}_recall@{k}'] = []
            aggregates[f'{source}_ndcg@{k}'] = []
        for alpha in fusion_alphas:
            tag = _alpha_tag(alpha)
            aggregates[f'fusion_{tag}_recall@{k}'] = []
            aggregates[f'fusion_{tag}_ndcg@{k}'] = []
    per_order_aggregates = [
        {
            **{f'recall@{k}': [] for k in metric_ks},
            **{f'ndcg@{k}': [] for k in metric_ks},
        }
        for _ in decode_orders
    ]
    with torch.no_grad():
        for batch in tqdm(loader, desc='Diffusion proposal + AR verification'):
            labels = batch['labels'].to(device)
            proposal_sets = []
            proposal_scores = []
            # All proposal orders condition on the same history.  Encode it
            # once instead of repeating the encoder pass for every order.
            encoder_hidden = diffusion_model(
                batch, return_loss=False
            ).hidden_states
            for decode_order in decode_orders:
                if args.proposal_mode == 'random':
                    diffusion_model.config['random_beam']['decode_order'] = decode_order
                candidates, scores = diffusion_model.generate(
                    batch,
                    n_return_sequences=args.proposal_k,
                    mode=args.proposal_mode,
                    return_scores=True,
                    encoder_hidden=encoder_hidden,
                )
                proposal_sets.append(candidates)
                proposal_scores.append(scores)
            proposals, valid_candidates, diffusion_scores = _deduplicate_candidates(
                proposal_sets, proposal_scores
            )
            diffusion_order = diffusion_scores.argsort(dim=1, descending=True)
            diffusion_ranked = proposals.gather(
                1, diffusion_order.unsqueeze(-1).expand(-1, -1, proposals.shape[-1])
            )
            path_scores = ar_model.score_candidate_paths(batch, proposals)
            path_scores = path_scores.masked_fill(~valid_candidates, float('-inf'))
            order = path_scores.argsort(dim=1, descending=True)
            verified = proposals.gather(
                1, order.unsqueeze(-1).expand(-1, -1, proposals.shape[-1])
            )
            diffusion_normalized = _normalize_candidate_scores(
                diffusion_scores, valid_candidates
            )
            ar_normalized = _normalize_candidate_scores(
                path_scores, valid_candidates
            )
            fused_rankings = {}
            for alpha in fusion_alphas:
                # Avoid 0 * (-inf) -> NaN at the alpha endpoints. Invalid
                # candidates are masked again after combining the two sources.
                fused_scores = (
                    (1.0 - alpha)
                    * diffusion_normalized.masked_fill(~valid_candidates, 0.0)
                    + alpha * ar_normalized.masked_fill(~valid_candidates, 0.0)
                ).masked_fill(~valid_candidates, float('-inf'))
                fused_order = fused_scores.argsort(dim=1, descending=True)
                fused_rankings[alpha] = proposals.gather(
                    1,
                    fused_order.unsqueeze(-1).expand(-1, -1, proposals.shape[-1]),
                )

            proposal_recall, _ = _ranking_metrics(
                proposals, labels, proposals.shape[1]
            )
            # "Diffusion" here means the first requested proposal order; the
            # multi-order union is evaluated separately through proposal recall
            # and AR-verified ranking.
            aggregates['proposal_recall'].extend(
                proposal_recall.detach().cpu().tolist()
            )
            for k in metric_ks:
                diff_recall, diff_ndcg = _ranking_metrics(
                    proposal_sets[0], labels, k
                )
                union_recall, union_ndcg = _ranking_metrics(
                    diffusion_ranked, labels, k
                )
                verified_recall, verified_ndcg = _ranking_metrics(
                    verified, labels, k
                )
                for key, values in (
                    (f'diffusion_recall@{k}', diff_recall),
                    (f'diffusion_ndcg@{k}', diff_ndcg),
                    (f'diffusion_union_recall@{k}', union_recall),
                    (f'diffusion_union_ndcg@{k}', union_ndcg),
                    (f'verified_recall@{k}', verified_recall),
                    (f'verified_ndcg@{k}', verified_ndcg),
                ):
                    aggregates[key].extend(values.detach().cpu().tolist())
                for alpha, fused in fused_rankings.items():
                    fused_recall, fused_ndcg = _ranking_metrics(fused, labels, k)
                    tag = _alpha_tag(alpha)
                    aggregates[f'fusion_{tag}_recall@{k}'].extend(
                        fused_recall.detach().cpu().tolist()
                    )
                    aggregates[f'fusion_{tag}_ndcg@{k}'].extend(
                        fused_ndcg.detach().cpu().tolist()
                    )
            aggregates['unique_candidates'].extend(
                valid_candidates.sum(dim=1).detach().cpu().tolist()
            )
            for order_idx, candidates in enumerate(proposal_sets):
                for k in metric_ks:
                    order_recall, order_ndcg = _ranking_metrics(
                        candidates, labels, k
                    )
                    per_order_aggregates[order_idx][f'recall@{k}'].extend(
                        order_recall.detach().cpu().tolist()
                    )
                    per_order_aggregates[order_idx][f'ndcg@{k}'].extend(
                        order_ndcg.detach().cpu().tolist()
                    )

    report = {
        'split': args.split,
        'catalog_sha256': digest,
        'catalog_n_items': n_catalog_items,
        'catalog_n_unique_sids': n_unique_sids,
        'catalog_collision_excess_ratio': (
            1.0 - n_unique_sids / n_catalog_items if n_catalog_items else 0.0
        ),
        'ar_checkpoint_sha256': ar_checkpoint_sha256,
        'diffusion_checkpoint_sha256': diffusion_checkpoint_sha256,
        'n_examples': len(aggregates['proposal_recall']),
        'decode_orders': decode_orders,
        'proposal_mode': args.proposal_mode,
        'metric_ks': metric_ks,
        'proposal_k_per_order': args.proposal_k,
        'fusion_alphas': fusion_alphas,
        'mean_unique_candidates': float(np.mean(aggregates['unique_candidates'])),
        'proposal_union_recall': float(np.mean(aggregates['proposal_recall'])),
        'per_order': [
            {
                'order': order,
                **{
                    f'recall@{k}': float(np.mean(values[f'recall@{k}']))
                    for k in metric_ks
                },
                **{
                    f'hit@{k}': float(np.mean(values[f'recall@{k}']))
                    for k in metric_ks
                },
                **{
                    f'ndcg@{k}': float(np.mean(values[f'ndcg@{k}']))
                    for k in metric_ks
                },
            }
            for order, values in zip(decode_orders, per_order_aggregates)
        ],
    }
    for k in metric_ks:
        for source, report_prefix in (
            ('diffusion', 'diffusion'),
            ('diffusion_union', 'diffusion_union_logsumexp'),
            ('verified', 'verified'),
        ):
            recall = float(np.mean(aggregates[f'{source}_recall@{k}']))
            report[f'{report_prefix}_recall@{k}'] = recall
            report[f'{report_prefix}_hit@{k}'] = recall
            report[f'{report_prefix}_ndcg@{k}'] = float(
                np.mean(aggregates[f'{source}_ndcg@{k}'])
            )
        for alpha in fusion_alphas:
            tag = _alpha_tag(alpha)
            recall = float(np.mean(aggregates[f'fusion_{tag}_recall@{k}']))
            report[f'fusion_alpha_{tag}_recall@{k}'] = recall
            report[f'fusion_alpha_{tag}_hit@{k}'] = recall
            report[f'fusion_alpha_{tag}_ndcg@{k}'] = float(
                np.mean(aggregates[f'fusion_{tag}_ndcg@{k}'])
            )
    if 10 in metric_ks:
        for alpha in fusion_alphas:
            tag = _alpha_tag(alpha)
            report[f'fusion_alpha_{tag}_weighted_score'] = (
                0.8 * report[f'fusion_alpha_{tag}_ndcg@10']
                + 0.2 * report[f'fusion_alpha_{tag}_recall@10']
            )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
