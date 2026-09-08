#!/usr/bin/env python
"""Train a Domino-style causal correction head over a frozen OPQ drafter.

The canonical parallel backbone and pairwise catalog selector are frozen.  A
small prefix GRU corrects their coordinate logits under teacher forcing.  At
inference, corrected path probabilities rerank a wide legal-item pool before
the existing AR verifier sees the final proposal budget.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.parallel_drafter import (
    CausalResidualCorrector,
    PairwisePathSelector,
    code_rows,
)
from genrec.utils import get_dataset
from scripts.train_parallel_opq_drafter import (
    encode_history,
    limit_dataset,
    make_config,
    normalize_scores,
    one_pass_decode,
    one_pass_outputs,
    ranking_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument(
        '--ar-checkpoint', default=None,
        help='Defaults to the AR checkpoint recorded by the base run.',
    )
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--min-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--residual-l2', type=float, default=1e-5)
    parser.add_argument(
        '--token-nll-weight', type=float, default=1.0,
        help='Weight of the original ground-truth token NLL.',
    )
    parser.add_argument(
        '--candidate-listwise-weight', type=float, default=0.0,
        help=(
            'Listwise CE weight on target-in-pool examples. The target is '
            'contrasted with the highest-scoring frozen-base negatives.'
        ),
    )
    parser.add_argument(
        '--ar-residual-distill-weight', type=float, default=0.0,
        help=(
            'Smooth-L1 weight for the normalized residual z(AR)-z(base). '
            'Unlike plain AR distillation, the frozen base remains the anchor.'
        ),
    )
    parser.add_argument('--distill-temperature', type=float, default=1.0)
    parser.add_argument(
        '--teacher-fusion-alpha', type=float, default=0.75,
        help='AR residual strength used by the training-time listwise score.',
    )
    parser.add_argument(
        '--train-candidates', type=int, default=32,
        help='Target/top-base hard-negative candidate-set width.',
    )
    parser.add_argument(
        '--boundary-hard-fraction', type=float, default=0.0,
        help=(
            'Fraction of training negatives drawn from a narrow band around '
            'the proposal-K boundary; the rest are frozen-base top negatives.'
        ),
    )
    parser.add_argument(
        '--hard-boundary-rank', type=int, default=None,
        help=(
            'Frozen-base rank around which boundary negatives are sampled. '
            'Defaults to proposal-k; set independently for a handoff-zone test.'
        ),
    )
    parser.add_argument(
        '--rank-aware-topk-weight', type=float, default=0.0,
        help=(
            'Soft boundary-loss weight for targets whose frozen-base ranks '
            'fall in the configured handoff interval. The target only needs '
            'to beat the K-th candidate negative, rather than become top-1.'
        ),
    )
    parser.add_argument('--rank-aware-min-rank', type=int, default=11)
    parser.add_argument('--rank-aware-max-rank', type=int, default=32)
    parser.add_argument('--rank-aware-top-k', type=int, default=10)
    parser.add_argument(
        '--head-preservation-weight', type=float, default=0.0,
        help=(
            'Squared-residual penalty on frozen-base head negatives for the '
            'rank-aware rows, preserving their internal top-K ordering.'
        ),
    )
    parser.add_argument('--head-preservation-k', type=int, default=10)
    parser.add_argument('--state-dim', type=int, default=64)
    parser.add_argument('--correction-rank', type=int, default=32)
    parser.add_argument(
        '--position-gamma', type=float, default=0.0,
        help=(
            'If positive, weight coordinate d by exp(-d/gamma), as in '
            'Domino. Zero uses uniform weights, the recommendation default.'
        ),
    )
    parser.add_argument('--pool-k', type=int, default=256)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument('--candidate-chunk-size', type=int, default=32)
    parser.add_argument('--ar-chunk-size', type=int, default=16)
    parser.add_argument('--correction-betas', default='0,0.25,0.5,0.75,1')
    parser.add_argument('--fusion-alphas', default='0,0.25,0.5,0.75,0.9,1')
    parser.add_argument(
        '--selection-metric',
        choices=('final_ndcg10', 'final_recall10', 'drafter_ndcg10'),
        default='final_ndcg10',
    )
    parser.add_argument(
        '--recall72-tolerance', type=float, default=None,
        help=(
            'If set, only select correction strengths whose proposal recall '
            'is at least beta=0 recall minus this absolute tolerance.'
        ),
    )
    parser.add_argument(
        '--epoch-selection-use-ar', action='store_true',
        help=(
            'Run the wide-pool AR verifier after every epoch. By default, '
            'epochs are selected by corrected drafter NDCG and AR is run once '
            'on the best head for full validation beta/alpha selection.'
        ),
    )
    parser.add_argument('--skip-ar-verifier', action='store_true')
    parser.add_argument('--max-train-examples', type=int, default=None)
    parser.add_argument('--max-val-examples', type=int, default=None)
    parser.add_argument('--max-test-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def parse_grid(text, name, lower=0.0, upper=1.0):
    values = tuple(dict.fromkeys(float(value) for value in text.split(',')))
    if not values or any(value < lower or value > upper for value in values):
        raise ValueError(f'{name} must contain values in [{lower},{upper}]')
    return values


def value_tag(value):
    return f'{float(value):g}'.replace('.', 'p')


def add_metrics(aggregates, prefix, ranked_codes, labels, cutoffs):
    for name, values in ranking_metrics(
        ranked_codes, labels, cutoffs=cutoffs
    ).items():
        aggregates[prefix + name] = (
            aggregates.get(prefix + name, 0.0)
            + float(values.float().sum().cpu())
        )


def gather_candidates(values, indices):
    if values.ndim == 2:
        return values.gather(1, indices)
    return values.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    )


def hard_candidate_batch(
    base_scores,
    target_rows,
    catalog,
    pool_k,
    n_candidates,
    proposal_k=None,
    boundary_fraction=0.0,
):
    """Build honest wide-pool positives plus frozen-base hard negatives.

    Target supervision is enabled only when the target was genuinely retrieved
    by the frozen base top-``pool_k``.  It is placed first for convenient
    listwise labels, but never turns an out-of-pool target into a positive.
    Rows without a pool positive still provide AR-residual distillation on the
    frozen base's top candidates.
    """
    pool_k = min(int(pool_k), int(base_scores.shape[1]))
    n_candidates = min(int(n_candidates), int(base_scores.shape[1]))
    if pool_k < 1 or n_candidates < 2:
        raise ValueError('pool_k >= 1 and n_candidates >= 2 are required')

    pool_rows = torch.topk(base_scores, k=pool_k, dim=1).indices
    target_in_pool = pool_rows.eq(target_rows[:, None]).any(dim=1)

    negative_scores = base_scores.clone()
    negative_scores.scatter_(1, target_rows[:, None], float('-inf'))
    boundary_count = int(round((n_candidates - 1) * float(boundary_fraction)))
    if boundary_count < 0 or boundary_count >= n_candidates:
        raise ValueError('boundary_fraction must produce 0..C-1 negatives')
    head_positive_count = n_candidates - 1 - boundary_count
    head_absent_count = n_candidates - boundary_count

    positive_parts = [target_rows[:, None]]
    absent_parts = []
    if head_positive_count:
        positive_parts.append(
            torch.topk(
                negative_scores, k=head_positive_count, dim=1
            ).indices
        )
    if head_absent_count:
        absent_parts.append(
            torch.topk(base_scores, k=head_absent_count, dim=1).indices
        )
    if boundary_count:
        if proposal_k is None:
            raise ValueError('proposal_k is required for boundary negatives')
        band_width = boundary_count + 1
        band_start = min(
            max(head_absent_count, int(proposal_k) - boundary_count // 2),
            pool_k - band_width,
        )
        if band_start < head_absent_count or band_start < 0:
            raise ValueError(
                'pool/proposal/train candidate sizes leave no disjoint '
                'boundary-negative band'
            )
        band_rows = pool_rows[:, band_start:band_start + band_width]
        band_scores = base_scores.gather(1, band_rows).masked_fill(
            band_rows.eq(target_rows[:, None]), float('-inf')
        )
        boundary_indices = torch.topk(
            band_scores, k=boundary_count, dim=1
        ).indices
        boundary_rows = band_rows.gather(1, boundary_indices)
        positive_parts.append(boundary_rows)
        absent_parts.append(boundary_rows)

    positive_first_rows = torch.cat(positive_parts, dim=1)
    base_top_rows = torch.cat(absent_parts, dim=1)
    if positive_first_rows.shape[1] != n_candidates:
        raise RuntimeError('positive candidate-set width mismatch')
    if base_top_rows.shape[1] != n_candidates:
        raise RuntimeError('fallback candidate-set width mismatch')
    candidate_rows = torch.where(
        target_in_pool[:, None], positive_first_rows, base_top_rows
    )
    return (
        catalog[candidate_rows],
        base_scores.gather(1, candidate_rows),
        candidate_rows,
        target_in_pool,
    )


def standardized_residuals(
    base_scores,
    ar_scores,
    student_residual,
    temperature=1.0,
):
    """Put teacher and student residuals on the frozen-base score scale."""
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    base_centered = base_scores - base_scores.mean(dim=1, keepdim=True)
    base_scale = base_centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    base_z = base_centered / base_scale
    ar_centered = ar_scores - ar_scores.mean(dim=1, keepdim=True)
    ar_scale = ar_centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    ar_z = ar_centered / ar_scale
    student_z = (
        student_residual - student_residual.mean(dim=1, keepdim=True)
    ) / base_scale
    return base_z / temperature, ar_z / temperature, student_z / temperature


def frozen_target_ranks(base_scores, target_rows):
    """Return deterministic one-based target ranks under frozen-base scores."""
    target_scores = base_scores.gather(1, target_rows[:, None])
    return base_scores.gt(target_scores).sum(dim=1) + 1


def rank_aware_topk_objective(
    student_scores,
    student_residual,
    target_ranks,
    target_in_pool,
    min_rank=11,
    max_rank=32,
    target_top_k=10,
    preserve_head_k=10,
):
    """Promote handoff-zone targets to Top-K without forcing them to rank 1.

    Candidate column zero must contain the genuine target for rows selected by
    ``target_in_pool``.  Subsequent leading columns are frozen-base head
    negatives, followed by optional proposal-boundary negatives.
    """
    eligible = (
        target_in_pool
        & target_ranks.ge(int(min_rank))
        & target_ranks.le(int(max_rank))
    )
    zero = student_scores.new_zeros(())
    if not eligible.any():
        return zero, zero, eligible

    negative_scores = student_scores[eligible, 1:]
    target_top_k = int(target_top_k)
    if target_top_k < 1 or target_top_k > negative_scores.shape[1]:
        raise ValueError('rank-aware-top-k exceeds available candidate negatives')
    topk_boundary = torch.topk(
        negative_scores, k=target_top_k, dim=1
    ).values[:, -1]
    promotion_loss = F.softplus(
        topk_boundary - student_scores[eligible, 0]
    ).mean()

    preserve_head_k = int(preserve_head_k)
    if preserve_head_k < 1 or preserve_head_k > student_residual.shape[1] - 1:
        raise ValueError(
            'head-preservation-k exceeds available candidate negatives'
        )
    preservation_loss = student_residual[
        eligible, 1:1 + preserve_head_k
    ].square().mean()
    return promotion_loss, preservation_loss, eligible


def corrected_pool_path_scores(
    corrector,
    hidden,
    logits,
    pool_codes,
    chunk_size,
):
    chunks = []
    for start in range(0, pool_codes.shape[1], int(chunk_size)):
        codes = pool_codes[:, start:start + int(chunk_size)]
        chunks.append(
            corrector(
                hidden,
                logits,
                codes,
                return_logits=False,
            )['path_log_probs']
        )
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def evaluate(
    model,
    selector,
    corrector,
    loader,
    catalog,
    pool_k,
    proposal_k,
    correction_betas,
    ar_model=None,
    fusion_alphas=(),
    candidate_chunk_size=32,
    ar_chunk_size=16,
    description='evaluation',
):
    model.eval()
    selector.eval()
    corrector.eval()
    if ar_model is not None:
        ar_model.eval()
    pool_k = min(int(pool_k), int(catalog.shape[0]))
    proposal_k = min(int(proposal_k), pool_k)
    aggregates = {}
    n_examples = 0
    started = time.perf_counter()

    for batch in tqdm(loader, desc=description):
        labels = batch['labels'].to(catalog.device)
        encoder_hidden = encode_history(model, batch)
        base_scores, logits, hidden, unary_scores, _ = one_pass_outputs(
            model,
            batch,
            catalog,
            selector,
            encoder_hidden=encoder_hidden,
        )
        pool_scores, pool_rows = torch.topk(
            base_scores, k=pool_k, dim=1
        )
        pool_codes = catalog[pool_rows]
        unary_pool_scores = unary_scores.gather(1, pool_rows)
        corrected_path_scores = corrected_pool_path_scores(
            corrector,
            hidden,
            logits,
            pool_codes,
            candidate_chunk_size,
        )
        add_metrics(
            aggregates,
            'pool_',
            pool_codes,
            labels,
            (pool_k,),
        )

        # Score the wide pool once, then gather the candidates selected by
        # every correction beta.  This is cheaper than rerunning the AR model
        # independently for each validation configuration.
        ar_pool_scores = None
        # A validation grid usually makes one wide-pool AR pass cheaper than
        # one final-budget pass per beta.  Fixed test evaluation normally has
        # only baseline + selected beta, for which direct scoring is cheaper.
        score_wide_ar_pool = (
            ar_model is not None
            and len(correction_betas) * proposal_k >= pool_k
        )
        if score_wide_ar_pool:
            ar_pool_scores = ar_model.score_candidate_paths(
                batch, pool_codes, chunk_size=ar_chunk_size
            )

        for beta in correction_betas:
            beta = float(beta)
            rerank_scores = pool_scores + beta * (
                corrected_path_scores - unary_pool_scores
            )
            proposal_scores, within_pool = torch.topk(
                rerank_scores, k=proposal_k, dim=1
            )
            proposals = gather_candidates(pool_codes, within_pool)
            beta_prefix = f'domino_b{value_tag(beta)}_'
            add_metrics(
                aggregates,
                beta_prefix,
                proposals,
                labels,
                (5, 10, proposal_k),
            )

            if ar_model is None:
                continue
            ar_scores = (
                gather_candidates(ar_pool_scores, within_pool)
                if ar_pool_scores is not None
                else ar_model.score_candidate_paths(
                    batch, proposals, chunk_size=ar_chunk_size
                )
            )
            ar_order = ar_scores.argsort(dim=1, descending=True)
            ar_ranked = gather_candidates(proposals, ar_order)
            add_metrics(
                aggregates,
                f'ar_b{value_tag(beta)}_',
                ar_ranked,
                labels,
                (5, 10),
            )
            for alpha in fusion_alphas:
                fused_scores = (
                    (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                    + float(alpha) * normalize_scores(ar_scores)
                )
                fused_order = fused_scores.argsort(dim=1, descending=True)
                fused_ranked = gather_candidates(proposals, fused_order)
                add_metrics(
                    aggregates,
                    (
                        f'fused_b{value_tag(beta)}_'
                        f'a{value_tag(alpha)}_'
                    ),
                    fused_ranked,
                    labels,
                    (5, 10),
                )
        n_examples += int(labels.shape[0])

    elapsed = time.perf_counter() - started
    result = {
        name: total / max(n_examples, 1)
        for name, total in aggregates.items()
    }
    result.update(
        n_examples=n_examples,
        elapsed_seconds=elapsed,
        milliseconds_per_example=1000.0 * elapsed / max(n_examples, 1),
        backbone_passes=1,
        correction_pool_k=pool_k,
        proposal_k=proposal_k,
    )
    return result


def select_configuration(
    metrics,
    betas,
    alphas,
    selection_metric,
    has_ar,
    recall72_tolerance=None,
):
    choices = []
    proposal_k = int(metrics['proposal_k'])
    baseline_recall = metrics[f'domino_b0_recall@{proposal_k}']
    for beta in betas:
        beta_tag = value_tag(beta)
        proposal_recall = metrics[
            f'domino_b{beta_tag}_recall@{proposal_k}'
        ]
        guard_pass = (
            recall72_tolerance is None
            or proposal_recall >= baseline_recall - float(recall72_tolerance)
        )
        if not guard_pass:
            continue
        if has_ar:
            for alpha in alphas:
                prefix = f'fused_b{beta_tag}_a{value_tag(alpha)}_'
                key = (
                    prefix + 'recall@10'
                    if selection_metric == 'final_recall10'
                    else prefix + 'ndcg@10'
                )
                # Break an exact NDCG tie using Recall, then prefer a smaller
                # correction to avoid needless deviation from the base.
                choices.append((
                    metrics[key],
                    metrics[prefix + 'recall@10'],
                    proposal_recall,
                    -float(beta),
                    float(beta),
                    float(alpha),
                    key,
                ))
        else:
            prefix = f'domino_b{beta_tag}_'
            key = prefix + (
                'ndcg@10'
                if selection_metric != 'final_recall10'
                else 'recall@10'
            )
            choices.append((
                metrics[key],
                metrics[prefix + 'recall@10'],
                proposal_recall,
                -float(beta),
                float(beta),
                None,
                key,
            ))
    selected = max(choices)
    return {
        'value': float(selected[0]),
        'recall_at_10': float(selected[1]),
        f'recall_at_{proposal_k}': float(selected[2]),
        f'baseline_recall_at_{proposal_k}': float(baseline_recall),
        'recall_guard_tolerance': recall72_tolerance,
        'beta': float(selected[4]),
        'alpha': selected[5],
        'metric_key': selected[6],
    }


def main():
    from accelerate import Accelerator

    args = parse_args()
    if args.epochs < 0 or args.patience < 1 or args.min_epochs < 0:
        raise ValueError('invalid epoch/patience settings')
    if args.pool_k < args.proposal_k or args.proposal_k < 1:
        raise ValueError('pool-k must be at least proposal-k >= 1')
    hard_boundary_rank = (
        args.proposal_k
        if args.hard_boundary_rank is None
        else args.hard_boundary_rank
    )
    if not 1 <= hard_boundary_rank <= args.pool_k:
        raise ValueError('--hard-boundary-rank must fall within the pool')
    if args.candidate_chunk_size < 1 or args.ar_chunk_size < 1:
        raise ValueError('candidate chunk sizes must be positive')
    if args.residual_l2 < 0 or args.position_gamma < 0:
        raise ValueError('regularization/gamma must be non-negative')
    for name in (
        'token_nll_weight',
        'candidate_listwise_weight',
        'ar_residual_distill_weight',
        'rank_aware_topk_weight',
        'head_preservation_weight',
    ):
        if getattr(args, name) < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be non-negative')
    if args.distill_temperature <= 0:
        raise ValueError('--distill-temperature must be positive')
    if not 0 <= args.teacher_fusion_alpha <= 1:
        raise ValueError('--teacher-fusion-alpha must be in [0,1]')
    if args.train_candidates < 2:
        raise ValueError('--train-candidates must be at least 2')
    if not 1 <= args.rank_aware_min_rank <= args.rank_aware_max_rank:
        raise ValueError('invalid rank-aware target-rank interval')
    if (
        args.rank_aware_topk_weight
        and not 1 <= args.rank_aware_top_k < args.train_candidates
    ):
        raise ValueError('--rank-aware-top-k must be below train-candidates')
    if (
        args.head_preservation_weight
        and not 1 <= args.head_preservation_k < args.train_candidates
    ):
        raise ValueError('--head-preservation-k must be below train-candidates')
    if not 0 <= args.boundary_hard_fraction < 1:
        raise ValueError('--boundary-hard-fraction must be in [0,1)')
    boundary_count = int(round(
        (args.train_candidates - 1) * args.boundary_hard_fraction
    ))
    head_negative_count = args.train_candidates - 1 - boundary_count
    if (
        args.rank_aware_topk_weight
        and head_negative_count < args.rank_aware_top_k
    ):
        raise ValueError(
            'boundary sampling leaves fewer head negatives than '
            '--rank-aware-top-k'
        )
    if (
        args.head_preservation_weight
        and head_negative_count < args.head_preservation_k
    ):
        raise ValueError(
            'boundary sampling leaves fewer head negatives than '
            '--head-preservation-k'
        )
    if args.recall72_tolerance is not None and args.recall72_tolerance < 0:
        raise ValueError('--recall72-tolerance must be non-negative')
    betas = parse_grid(args.correction_betas, 'correction-betas')
    alphas = parse_grid(args.fusion_alphas, 'fusion-alphas')
    if 0.0 not in betas:
        raise ValueError('correction-betas must include 0 for the frozen-base guard')
    if args.skip_ar_verifier and args.selection_metric.startswith('final_'):
        raise ValueError(
            'final_* selection requires AR verification; use drafter_ndcg10'
        )
    if args.skip_ar_verifier and args.ar_residual_distill_weight:
        raise ValueError('AR residual distillation requires the AR verifier')
    if args.skip_ar_verifier and args.epoch_selection_use_ar:
        raise ValueError('--epoch-selection-use-ar requires the AR verifier')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_checkpoint_path = Path(args.base_checkpoint).resolve()
    base_checkpoint = torch.load(
        base_checkpoint_path, map_location='cpu', weights_only=False
    )
    base_args = base_checkpoint['args']
    required = (
        'dataset', 'common_config', 'ar_config', 'diffusion_config'
    )
    missing = [name for name in required if not base_args.get(name)]
    if missing:
        raise ValueError(f'base checkpoint is missing arguments: {missing}')
    if base_checkpoint.get('selector') is None:
        raise ValueError('Domino reranking requires a pairwise base selector')
    if base_args.get('backbone_architecture', 'masked_decoder') != 'masked_decoder':
        raise ValueError('this experiment currently expects a masked decoder base')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()
    common_files = [base_args['common_config']]
    if base_args.get('sid_config'):
        common_files.append(base_args['sid_config'])
    diffusion_config = make_config(
        'DIFF_GRM',
        base_args['dataset'],
        common_files + [base_args['diffusion_config']],
        accelerator,
        {
            'train_batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
        },
    )
    ar_config = make_config(
        'AR_GRM',
        base_args['dataset'],
        common_files + [base_args['ar_config']],
        accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    device = torch.device(diffusion_config['device'])
    dataset = get_dataset(base_args['dataset'])(diffusion_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    raw_splits = dataset.split()
    raw_splits['train'] = limit_dataset(
        raw_splits['train'], args.max_train_examples
    )
    raw_splits['val'] = limit_dataset(
        raw_splits['val'], args.max_val_examples
    )
    raw_splits['test'] = limit_dataset(
        raw_splits['test'], args.max_test_examples
    )
    tokenized = tokenizer.tokenize(raw_splits)
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('causal reranking requires collision-free catalog SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        tokenized['train'],
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=tokenizer.collate_fn['train'],
    )
    val_loader = DataLoader(
        tokenized['val'],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        tokenized['test'],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )

    model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    model.load_state_dict(base_checkpoint['model'])
    pair_rank = int(base_checkpoint['selector']['left'].shape[-1])
    triple_rank = (
        int(base_checkpoint['selector']['triple_factors'].shape[-1])
        if base_checkpoint['selector'].get('triple_factors') is not None
        else 0
    )
    selector = PairwisePathSelector(
        model.n_digit,
        model.codebook_size,
        model.n_embd,
        rank=pair_rank,
        triple_rank=triple_rank,
    ).to(device)
    selector.load_state_dict(base_checkpoint['selector'])
    model.eval()
    selector.eval()
    for module in (model, selector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    torch.manual_seed(args.seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 2)
    corrector = CausalResidualCorrector(
        model.n_digit,
        model.codebook_size,
        model.n_embd,
        state_dim=args.state_dim,
        rank=args.correction_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    ar_model = None
    ar_checkpoint_path = args.ar_checkpoint or base_args.get('ar_checkpoint')
    if not args.skip_ar_verifier:
        if not ar_checkpoint_path:
            raise ValueError('no AR checkpoint was provided or recorded')
        ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
        ar_model.load_state_dict(
            torch.load(ar_checkpoint_path, map_location=device)
        )
        ar_model.eval()
        for parameter in ar_model.parameters():
            parameter.requires_grad_(False)

    if args.position_gamma:
        positions = torch.arange(
            model.n_digit, device=device, dtype=torch.float32
        )
        position_weights = torch.exp(-positions / args.position_gamma)
    else:
        position_weights = torch.ones(model.n_digit, device=device)
    position_weights = position_weights / position_weights.sum()

    checkpoint_path = output_dir / 'best.pt'
    history = []
    best_value = float('-inf')
    no_improve = 0

    for epoch in range(0, args.epochs + 1):
        train_summary = {}
        if epoch:
            corrector.train()
            loss_sum = 0.0
            nll_sum = 0.0
            listwise_sum = 0.0
            distill_sum = 0.0
            rank_aware_sum = 0.0
            head_preservation_sum = 0.0
            residual_sum = 0.0
            pool_positive_sum = 0
            rank_aware_example_sum = 0
            example_sum = 0
            base_top1_sum = 0
            student_top1_sum = 0
            teacher_top1_sum = 0
            n_batches = 0
            for batch in tqdm(train_loader, desc=f'train correction epoch {epoch}'):
                optimizer.zero_grad(set_to_none=True)
                targets = batch['decoder_labels'].to(device)
                with torch.no_grad():
                    encoder_hidden = encode_history(model, batch)
                    (
                        base_scores,
                        logits,
                        hidden,
                        unary_scores,
                        _,
                    ) = one_pass_outputs(
                        model,
                        batch,
                        catalog,
                        selector,
                        encoder_hidden=encoder_hidden,
                    )

                target_output = corrector(hidden, logits, targets)
                nll = -(
                    target_output['token_log_probs'][:, 0]
                    * position_weights[None]
                ).sum(dim=1).mean()
                residual_penalty = (
                    target_output['residual_logits'].float().square().mean()
                )

                listwise_loss = nll.new_zeros(())
                distill_loss = nll.new_zeros(())
                rank_aware_loss = nll.new_zeros(())
                head_preservation_loss = nll.new_zeros(())
                target_in_pool = torch.zeros(
                    targets.shape[0], dtype=torch.bool, device=device
                )
                rank_aware_mask = target_in_pool
                base_top1 = student_top1 = teacher_top1 = 0
                if (
                    args.candidate_listwise_weight
                    or args.ar_residual_distill_weight
                    or args.rank_aware_topk_weight
                    or args.head_preservation_weight
                ):
                    target_rows = code_rows(
                        targets, catalog, model.codebook_size
                    )
                    target_ranks = frozen_target_ranks(
                        base_scores.detach(), target_rows
                    )
                    (
                        candidates,
                        candidate_base_scores,
                        candidate_rows,
                        target_in_pool,
                    ) = hard_candidate_batch(
                        base_scores.detach(),
                        target_rows,
                        catalog,
                        args.pool_k,
                        args.train_candidates,
                        proposal_k=hard_boundary_rank,
                        boundary_fraction=args.boundary_hard_fraction,
                    )
                    candidate_unary_scores = unary_scores.gather(
                        1, candidate_rows
                    )
                    candidate_output = corrector(
                        hidden,
                        logits,
                        candidates,
                        return_logits=False,
                    )
                    student_raw_residual = (
                        candidate_output['path_log_probs']
                        - candidate_unary_scores
                    )
                    if args.ar_residual_distill_weight and ar_model is not None:
                        with torch.no_grad():
                            teacher_ar_scores = ar_model.score_candidate_paths(
                                batch,
                                candidates,
                                chunk_size=args.ar_chunk_size,
                            )
                    else:
                        teacher_ar_scores = candidate_base_scores
                    base_z, ar_z, student_z = standardized_residuals(
                        candidate_base_scores,
                        teacher_ar_scores,
                        student_raw_residual,
                        temperature=args.distill_temperature,
                    )
                    teacher_residual = (ar_z - base_z).detach()
                    distill_loss = F.smooth_l1_loss(
                        student_z, teacher_residual
                    )
                    student_training_scores = (
                        base_z
                        + float(args.teacher_fusion_alpha) * student_z
                    )
                    teacher_training_scores = (
                        base_z
                        + float(args.teacher_fusion_alpha) * teacher_residual
                    )
                    if (
                        args.rank_aware_topk_weight
                        or args.head_preservation_weight
                    ):
                        (
                            rank_aware_loss,
                            head_preservation_loss,
                            rank_aware_mask,
                        ) = rank_aware_topk_objective(
                            student_training_scores,
                            student_z,
                            target_ranks,
                            target_in_pool,
                            min_rank=args.rank_aware_min_rank,
                            max_rank=args.rank_aware_max_rank,
                            target_top_k=args.rank_aware_top_k,
                            preserve_head_k=args.head_preservation_k,
                        )
                    if target_in_pool.any():
                        positive_scores = student_training_scores[target_in_pool]
                        listwise_loss = F.cross_entropy(
                            positive_scores,
                            torch.zeros(
                                positive_scores.shape[0],
                                dtype=torch.long,
                                device=device,
                            ),
                        )
                        base_top1 = int(
                            base_z[target_in_pool].argmax(dim=1).eq(0).sum()
                        )
                        student_top1 = int(
                            positive_scores.argmax(dim=1).eq(0).sum()
                        )
                        teacher_top1 = int(
                            teacher_training_scores[target_in_pool]
                            .argmax(dim=1).eq(0).sum()
                        )

                loss = (
                    float(args.token_nll_weight) * nll
                    + float(args.candidate_listwise_weight) * listwise_loss
                    + float(args.ar_residual_distill_weight) * distill_loss
                    + float(args.rank_aware_topk_weight) * rank_aware_loss
                    + float(args.head_preservation_weight)
                    * head_preservation_loss
                    + float(args.residual_l2) * residual_penalty
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach())
                nll_sum += float(nll.detach())
                listwise_sum += float(listwise_loss.detach())
                distill_sum += float(distill_loss.detach())
                rank_aware_sum += float(rank_aware_loss.detach())
                head_preservation_sum += float(
                    head_preservation_loss.detach()
                )
                residual_sum += float(residual_penalty.detach())
                pool_positive_sum += int(target_in_pool.sum())
                rank_aware_example_sum += int(rank_aware_mask.sum())
                example_sum += int(targets.shape[0])
                base_top1_sum += base_top1
                student_top1_sum += student_top1
                teacher_top1_sum += teacher_top1
                n_batches += 1
            train_summary = {
                'train_loss': loss_sum / max(n_batches, 1),
                'train_nll': nll_sum / max(n_batches, 1),
                'train_candidate_listwise': listwise_sum / max(n_batches, 1),
                'train_ar_residual_distill': distill_sum / max(n_batches, 1),
                'train_rank_aware_topk': rank_aware_sum / max(n_batches, 1),
                'train_head_preservation': (
                    head_preservation_sum / max(n_batches, 1)
                ),
                'train_residual_l2': residual_sum / max(n_batches, 1),
                'train_pool_positive_rate': (
                    pool_positive_sum / max(example_sum, 1)
                ),
                'train_rank_aware_example_rate': (
                    rank_aware_example_sum / max(example_sum, 1)
                ),
                'train_base_hardset_top1': (
                    base_top1_sum / max(pool_positive_sum, 1)
                ),
                'train_student_hardset_top1': (
                    student_top1_sum / max(pool_positive_sum, 1)
                ),
                'train_teacher_hardset_top1': (
                    teacher_top1_sum / max(pool_positive_sum, 1)
                ),
            }

        epoch_ar_model = ar_model if args.epoch_selection_use_ar else None
        validation = evaluate(
            model,
            selector,
            corrector,
            val_loader,
            catalog,
            args.pool_k,
            args.proposal_k,
            betas,
            ar_model=epoch_ar_model,
            fusion_alphas=alphas if epoch_ar_model is not None else (),
            candidate_chunk_size=args.candidate_chunk_size,
            ar_chunk_size=args.ar_chunk_size,
            description=f'validation epoch {epoch}',
        )
        selection = select_configuration(
            validation,
            betas,
            alphas,
            args.selection_metric,
            has_ar=epoch_ar_model is not None,
            recall72_tolerance=args.recall72_tolerance,
        )
        validation.update(epoch=epoch, selection=selection, **train_summary)
        history.append(validation)
        print(
            json.dumps(
                {
                    'epoch': epoch,
                    'selection': selection,
                    'pool_recall': validation[
                        f'pool_recall@{validation["correction_pool_k"]}'
                    ],
                    **train_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if selection['value'] > best_value:
            best_value = selection['value']
            no_improve = 0
            torch.save(
                {
                    'corrector': corrector.state_dict(),
                    'args': vars(args),
                    'base_checkpoint': str(base_checkpoint_path),
                    'validation': validation,
                    'selection': selection,
                    'epoch': epoch,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if (
                epoch >= args.min_epochs
                and epoch < args.epochs
                and no_improve >= args.patience
            ):
                print(
                    f'[EARLY STOP] no validation improvement for {no_improve} epochs',
                    flush=True,
                )
                break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    corrector.load_state_dict(best['corrector'])
    validation_with_verifier = evaluate(
        model,
        selector,
        corrector,
        val_loader,
        catalog,
        args.pool_k,
        args.proposal_k,
        betas,
        ar_model=ar_model,
        fusion_alphas=alphas if ar_model is not None else (),
        candidate_chunk_size=args.candidate_chunk_size,
        ar_chunk_size=args.ar_chunk_size,
        description='validation beta/alpha selection',
    )
    final_selection = select_configuration(
        validation_with_verifier,
        betas,
        alphas,
        args.selection_metric,
        has_ar=ar_model is not None,
        recall72_tolerance=args.recall72_tolerance,
    )
    selected_beta = float(final_selection['beta'])
    selected_alpha = final_selection['alpha']
    test_betas = tuple(dict.fromkeys((0.0, selected_beta)))
    test_alphas = (
        tuple(dict.fromkeys((0.75, float(selected_alpha))))
        if selected_alpha is not None else ()
    )
    test = evaluate(
        model,
        selector,
        corrector,
        test_loader,
        catalog,
        args.pool_k,
        args.proposal_k,
        test_betas,
        ar_model=ar_model,
        fusion_alphas=test_alphas,
        candidate_chunk_size=args.candidate_chunk_size,
        ar_chunk_size=args.ar_chunk_size,
        description='test fixed validation selection',
    )
    report = {
        'protocol': vars(args),
        'base_checkpoint': str(base_checkpoint_path),
        'base_validation': base_checkpoint.get('validation'),
        'catalog_items': int(catalog.shape[0]),
        'corrector_parameters': sum(
            parameter.numel() for parameter in corrector.parameters()
        ),
        'position_weights': position_weights.detach().cpu().tolist(),
        'best_epoch': int(best['epoch']),
        'best_epoch_validation': best['validation'],
        'validation_with_verifier': validation_with_verifier,
        'selected_configuration': final_selection,
        'history': history,
        'test': test,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report['selected_configuration'], sort_keys=True))
    print(json.dumps(test, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
