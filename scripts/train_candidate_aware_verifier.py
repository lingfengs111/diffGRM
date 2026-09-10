#!/usr/bin/env python
"""Fine-tune an AR model as a proposal-aware item verifier.

The proposal model is frozen.  For every training history it retrieves hard
item negatives from the legal collision-free OPQ catalog in one parallel
pass.  The AR model is then optimized with its original token CE plus an item
listwise loss over the positive path and those proposal negatives.
"""

from __future__ import annotations

import argparse
import copy
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
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.encoder_head_drafter import (
    EncoderOnlyFourHeadDrafter,
)
from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector, code_rows
from genrec.utils import get_config, get_dataset
from scripts.train_parallel_opq_drafter import (
    evaluate,
    normalize_scores,
    one_pass_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2014CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--sid-config', default=None)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--diffusion-checkpoint', default=None)
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--pair-rank', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--min-epochs', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--token-loss-weight', type=float, default=1.0)
    parser.add_argument('--listwise-weight', type=float, required=True)
    parser.add_argument('--listwise-temperature', type=float, default=1.0)
    parser.add_argument('--margin-weight', type=float, default=0.0)
    parser.add_argument('--margin-value', type=float, default=0.2)
    parser.add_argument('--teacher-distill-weight', type=float, default=0.0)
    parser.add_argument('--teacher-temperature', type=float, default=1.0)
    parser.add_argument(
        '--prefix-rank-weight', type=float, default=0.0,
        help='APAO-style loss weight over every cumulative SID prefix.',
    )
    parser.add_argument('--prefix-rank-temperature', type=float, default=1.0)
    parser.add_argument('--prefix-adaptive-eta', type=float, default=0.0)
    parser.add_argument(
        '--prefix-negative-scale', action='store_true',
        help='Scale sampled prefix negatives to the number of legal catalog prefixes.',
    )
    parser.add_argument(
        '--preference-weight', type=float, default=0.0,
        help='Weight of reference-anchored pairwise preference optimization.',
    )
    parser.add_argument('--preference-beta', type=float, default=0.5)
    parser.add_argument('--preference-ndcg-k', type=int, default=10)
    parser.add_argument(
        '--ranking-score', choices=('fused', 'ar'), default='fused',
        help='Optimize the historical fused score or the verifier path score itself.',
    )
    parser.add_argument(
        '--require-target-in-proposals', action='store_true',
        help=(
            'Apply listwise/margin losses only when the frozen drafter really '
            'retrieves the positive inside proposal-k. This matches verifier '
            'inference and avoids learning from injected impossible positives.'
        ),
    )
    parser.add_argument(
        '--negative-strata', default=None,
        help=(
            'Comma-separated negative counts from proposal ranks 1-10, '
            '11-32, and 33-proposal_k, e.g. 8,4,3. The default keeps the '
            'historical top-hard-negative behavior.'
        ),
    )
    parser.add_argument(
        '--negative-mining', choices=('proposal', 'current_ar'),
        default='proposal',
        help=(
            'proposal uses drafter-rank negatives; current_ar re-scores the '
            'entire proposal pool with the current verifier and selects its '
            'highest-scoring mistakes every step.'
        ),
    )
    parser.add_argument('--head-rank-weight', type=float, default=1.0)
    parser.add_argument('--middle-rank-weight', type=float, default=1.0)
    parser.add_argument('--tail-rank-weight', type=float, default=1.0)
    parser.add_argument(
        '--training-fusion-alpha',
        type=float,
        default=0.75,
        help=(
            'AR weight used inside the residual verifier training loss. '
            'The proposal weight is 1-alpha; validation still selects from '
            'the full --fusion-alphas grid.'
        ),
    )
    parser.add_argument(
        '--trainable-scope',
        choices=('all', 'decoder', 'last_decoder'),
        default='last_decoder',
        help='Conservative verifier adaptation scope.',
    )
    parser.add_argument('--num-negatives', type=int, default=15)
    parser.add_argument('--candidate-score-chunk-size', type=int, default=4)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument(
        '--fusion-alphas', default='0,0.1,0.25,0.5,0.75,0.9,1'
    )
    parser.add_argument('--max-train-examples', type=int, default=None)
    parser.add_argument('--max-val-examples', type=int, default=None)
    parser.add_argument('--max-test-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def make_config(model, dataset, files, accelerator, overrides=None):
    config = get_config(model, dataset, files, overrides or {})
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator
    return config


def limit_dataset(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    return dataset.select(range(int(maximum)))


def proposal_negatives(
    drafter_model,
    selector,
    batch,
    catalog,
    targets,
    num_negatives,
    proposal_k,
    negative_strata=None,
    negative_mining='proposal',
    ar_model=None,
    candidate_score_chunk_size=8,
):
    """Return positive-first candidates sampled from the real proposal list.

    ``target_ranks`` are one-indexed inside the top-k list and ``k+1`` when
    absent.  The target remains column zero only as a convenient listwise
    label; callers can mask absent targets so training exactly reflects the
    verifier's inference support.
    """
    with torch.no_grad():
        proposal_scores, _, _, _, _ = one_pass_outputs(
            drafter_model, batch, catalog, selector
        )
        target_rows = code_rows(
            targets, catalog, drafter_model.codebook_size
        )
        proposal_keep = min(int(proposal_k), int(catalog.shape[0]))
        top_scores, top_rows = torch.topk(
            proposal_scores, k=proposal_keep, dim=1
        )
        target_matches = top_rows.eq(target_rows[:, None])
        recalled = target_matches.any(dim=1)
        target_ranks = torch.where(
            recalled,
            target_matches.float().argmax(dim=1).long() + 1,
            torch.full_like(target_rows, proposal_keep + 1),
        )
        keep = min(int(num_negatives), proposal_keep - 1)
        if negative_mining == 'current_ar':
            if ar_model is None:
                raise ValueError('current_ar mining requires ar_model')
            masked = top_scores.masked_fill(target_matches, float('-inf'))
            pool_scores, positions = torch.topk(
                masked, k=proposal_keep - 1, dim=1
            )
            pool_rows = top_rows.gather(1, positions)
            pool_codes = catalog[pool_rows]
            was_training = ar_model.training
            ar_model.eval()
            ar_pool_scores = ar_model.score_candidate_paths(
                batch,
                pool_codes,
                chunk_size=candidate_score_chunk_size,
            )
            ar_model.train(was_training)
            _, hard_positions = torch.topk(ar_pool_scores, k=keep, dim=1)
            hard_rows = pool_rows.gather(1, hard_positions)
            hard_scores = pool_scores.gather(1, hard_positions)
        elif negative_strata is None:
            masked = top_scores.masked_fill(target_matches, float('-inf'))
            hard_scores, positions = torch.topk(masked, k=keep, dim=1)
            hard_rows = top_rows.gather(1, positions)
        else:
            quotas = tuple(int(value) for value in negative_strata)
            if len(quotas) != 3 or any(value < 0 for value in quotas):
                raise ValueError('negative_strata must contain three non-negative counts')
            if sum(quotas) != keep:
                raise ValueError(
                    f'negative_strata sums to {sum(quotas)}, expected {keep}'
                )
            boundaries = ((0, min(10, proposal_keep)),
                          (min(10, proposal_keep), min(32, proposal_keep)),
                          (min(32, proposal_keep), proposal_keep))
            selected_rows = []
            selected_scores = []
            # Only 15--31 candidates are selected per example.  This small
            # loop makes the rank-stratified protocol explicit and avoids
            # silently collapsing back to top-hard negatives.
            target_positions = [
                rank - 1 if rank <= proposal_keep else -1
                for rank in target_ranks.detach().cpu().tolist()
            ]
            for batch_idx, target_position in enumerate(target_positions):
                row_positions = []
                for quota, (start, end) in zip(quotas, boundaries):
                    eligible = [
                        pos for pos in range(start, end)
                        if pos != target_position
                    ]
                    row_positions.extend(eligible[:quota])
                if len(row_positions) < keep:
                    chosen = set(row_positions)
                    fallback = [
                        pos for pos in range(proposal_keep)
                        if pos not in chosen
                        and pos != target_position
                    ]
                    row_positions.extend(fallback[:keep - len(row_positions)])
                position_tensor = torch.as_tensor(
                    row_positions[:keep], dtype=torch.long,
                    device=top_rows.device,
                )
                selected_rows.append(top_rows[batch_idx, position_tensor])
                selected_scores.append(top_scores[batch_idx, position_tensor])
            hard_rows = torch.stack(selected_rows, dim=0)
            hard_scores = torch.stack(selected_scores, dim=0)
        hard_codes = catalog[hard_rows]
        candidates = torch.cat([targets[:, None, :], hard_codes], dim=1)
        candidate_proposal_scores = torch.cat(
            [proposal_scores.gather(1, target_rows[:, None]), hard_scores], dim=1
        )
    return (
        candidates, candidate_proposal_scores, recalled, target_ranks
    )


def candidate_token_log_scores(model, batch, candidates, chunk_size):
    """Return differentiable log p(code_t | history, code_<t) per candidate."""
    history_sid = batch['history_sid'].to(next(model.parameters()).device)
    encoded = model({'history_sid': history_sid}, return_loss=False).hidden_states
    history_mask = history_sid.ne(-1).any(dim=-1)
    logits = model.candidate_logits_from_encoded_history(
        encoded,
        history_mask,
        candidates,
        chunk_size=chunk_size,
    )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)


def duplicate_prefix_mask(candidates, length):
    """Mask a candidate when an identical prefix appeared in an earlier column."""
    prefixes = candidates[:, :, :length]
    equal = prefixes[:, :, None, :].eq(prefixes[:, None, :, :]).all(dim=-1)
    earlier = torch.tril(
        torch.ones(
            equal.shape[1], equal.shape[2], dtype=torch.bool,
            device=equal.device,
        ),
        diagonal=-1,
    )
    return (equal & earlier.unsqueeze(0)).any(dim=-1)


def prefix_pairwise_loss(
    token_scores,
    candidates,
    example_weights,
    temperature,
    prefix_counts,
    negative_scale=False,
    previous_adaptive_weights=None,
    adaptive_eta=0.0,
):
    """APAO-style sampled-softmax loss on each unique cumulative prefix."""
    cumulative = token_scores.cumsum(dim=-1)
    losses = []
    for digit in range(token_scores.shape[-1]):
        scores = cumulative[:, :, digit] / float(temperature)
        duplicate = duplicate_prefix_mask(candidates, digit + 1)
        negative_scores = scores[:, 1:].masked_fill(
            duplicate[:, 1:], float('-inf')
        )
        if negative_scale:
            sampled = (~duplicate[:, 1:]).sum(dim=1).clamp_min(1)
            scale = (
                (float(prefix_counts[digit]) - 1.0)
                / sampled.to(scores.dtype)
            ).clamp_min(1.0)
            negative_scores = negative_scores + scale.log()[:, None]
        denominator = torch.logsumexp(
            torch.cat([scores[:, :1], negative_scores], dim=1), dim=1
        )
        per_example = (denominator - scores[:, 0]) / float(digit + 1)
        losses.append(weighted_mean(per_example, example_weights))
    losses = torch.stack(losses)
    detached = losses.detach().clamp_min(0)
    if float(detached.sum()) == 0.0:
        normalized = (
            torch.full_like(detached, 1.0 / len(detached))
            if previous_adaptive_weights is None
            else previous_adaptive_weights.to(detached)
        )
    else:
        normalized = detached / detached.sum()
    if previous_adaptive_weights is None:
        adaptive = normalized
    else:
        adaptive = previous_adaptive_weights.to(losses) * torch.exp(
            float(adaptive_eta) * normalized
        )
        adaptive = adaptive / adaptive.sum().clamp_min(1e-12)
    return (adaptive * losses).sum(), losses, adaptive.detach()


def ndcg_lambda_dpo_loss(
    path_scores,
    reference_scores,
    proposal_scores,
    example_weights,
    beta,
    ndcg_k,
    fusion_alpha,
):
    """Reference-relative positive-vs-negative DPO weighted by Delta-NDCG@K."""
    with torch.no_grad():
        fused = (
            (1.0 - float(fusion_alpha)) * normalize_scores(proposal_scores)
            + float(fusion_alpha) * normalize_scores(path_scores.detach())
        )
        order = fused.argsort(dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            1, fused.shape[1] + 1, device=fused.device
        ).unsqueeze(0).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        discounts = torch.where(
            ranks <= int(ndcg_k),
            1.0 / torch.log2(ranks.to(fused.dtype) + 1.0),
            torch.zeros_like(fused),
        )
        pair_weights = (discounts[:, :1] - discounts[:, 1:]).abs()
        pair_weights = pair_weights / pair_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
    student_gap = path_scores[:, :1] - path_scores[:, 1:]
    reference_gap = reference_scores[:, :1] - reference_scores[:, 1:]
    pair_losses = F.softplus(-float(beta) * (student_gap - reference_gap))
    per_example = (pair_weights * pair_losses).sum(dim=1)
    return weighted_mean(per_example, example_weights), pair_weights


def weighted_mean(values, weights):
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def rank_weights(target_ranks, args):
    return torch.where(
        target_ranks <= 10,
        torch.full_like(target_ranks, args.head_rank_weight, dtype=torch.float),
        torch.where(
            target_ranks <= 32,
            torch.full_like(
                target_ranks, args.middle_rank_weight, dtype=torch.float
            ),
            torch.full_like(
                target_ranks, args.tail_rank_weight, dtype=torch.float
            ),
        ),
    )


def validation_score(result, fusion_alphas, metric_k=10):
    scores = {}
    for alpha in fusion_alphas:
        tag = f'{float(alpha):g}'.replace('.', 'p')
        scores[float(alpha)] = result[f'fused_a{tag}_ndcg@{metric_k}']
    best_alpha = max(scores, key=scores.get)
    return float(scores[best_alpha]), float(best_alpha)


def configure_trainable_scope(model, scope):
    """Keep the strong generator intact while adapting it for verification."""
    if scope == 'all':
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        last_decoder = len(model.decoder_blocks) - 1
        for name, parameter in model.named_parameters():
            trainable = (
                name.startswith('decoder_blocks.')
                if scope == 'decoder'
                else name.startswith(f'decoder_blocks.{last_decoder}.')
            )
            if trainable or name.startswith('ln_f.') or name == 'bos_embedding':
                parameter.requires_grad_(True)
    selected = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not selected:
        raise ValueError(f'no trainable parameters for scope={scope}')
    return selected


def main():
    args = parse_args()
    if args.listwise_weight < 0:
        raise ValueError('--listwise-weight must be non-negative')
    if args.listwise_temperature <= 0:
        raise ValueError('--listwise-temperature must be positive')
    if args.margin_weight < 0 or args.margin_value < 0:
        raise ValueError('--margin-weight/value must be non-negative')
    if args.teacher_distill_weight < 0 or args.teacher_temperature <= 0:
        raise ValueError('--teacher-distill-weight must be non-negative and temperature positive')
    if args.prefix_rank_weight < 0 or args.prefix_rank_temperature <= 0:
        raise ValueError('--prefix-rank-weight must be non-negative and temperature positive')
    if args.prefix_adaptive_eta < 0:
        raise ValueError('--prefix-adaptive-eta must be non-negative')
    if args.preference_weight < 0 or args.preference_beta <= 0:
        raise ValueError('--preference-weight must be non-negative and beta positive')
    if args.preference_ndcg_k < 1:
        raise ValueError('--preference-ndcg-k must be positive')
    if min(args.head_rank_weight, args.middle_rank_weight, args.tail_rank_weight) < 0:
        raise ValueError('rank weights must be non-negative')
    if not 0.0 <= args.training_fusion_alpha <= 1.0:
        raise ValueError('--training-fusion-alpha must lie in [0,1]')
    if args.num_negatives < 1:
        raise ValueError('--num-negatives must be positive')
    negative_strata = None
    if args.negative_strata:
        negative_strata = tuple(
            int(value) for value in args.negative_strata.split(',')
        )
    if args.negative_mining == 'current_ar' and negative_strata is not None:
        raise ValueError('current_ar mining and negative_strata are mutually exclusive')
    fusion_alphas = [float(value) for value in args.fusion_alphas.split(',')]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()
    common_files = [args.common_config]
    if args.sid_config:
        common_files.append(args.sid_config)
    diffusion_config = make_config(
        'DIFF_GRM', args.dataset,
        common_files + [args.diffusion_config], accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    ar_config = make_config(
        'AR_GRM', args.dataset,
        common_files + [args.ar_config], accelerator,
        {
            'eval_batch_size': args.eval_batch_size,
            'candidate_score_chunk_size': args.candidate_score_chunk_size,
        },
    )
    device = torch.device(diffusion_config['device'])
    dataset = get_dataset(args.dataset)(diffusion_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    raw_splits = dataset.split()
    for split, maximum in (
        ('train', args.max_train_examples),
        ('val', args.max_val_examples),
        ('test', args.max_test_examples),
    ):
        raw_splits[split] = limit_dataset(raw_splits[split], maximum)
    tokenized = tokenizer.tokenize(raw_splits)
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('candidate-aware verification requires collision-free SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    train_loader = DataLoader(
        limit_dataset(tokenized['train'], args.max_train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tokenizer.collate_fn['train'],
    )
    val_loader = DataLoader(
        limit_dataset(tokenized['val'], args.max_val_examples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        limit_dataset(tokenized['test'], args.max_test_examples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )

    drafter_state = torch.load(args.drafter_checkpoint, map_location=device)
    checkpoint_args = drafter_state.get('args', {})
    backbone_architecture = checkpoint_args.get(
        'backbone_architecture', 'masked_decoder'
    )
    if backbone_architecture == 'encoder_four_head':
        diffusion_config['encoder_head_n_layer'] = int(
            checkpoint_args.get('encoder_head_n_layer', 4)
        )
        diffusion_config['encoder_head_normalize_logits'] = (
            checkpoint_args.get('training_objective', 'catalog_plus_token')
            == 'mtp_only'
        )
        diffusion_config['encoder_head_logit_temperature'] = float(
            checkpoint_args.get('mtp_temperature', 0.07)
        )
        diffusion_config.update(
            history_head=checkpoint_args.get('history_head', 'pooled'),
            n_interests=int(checkpoint_args.get('n_interests', 1)),
            interest_temperature=float(
                checkpoint_args.get('interest_temperature', 1.0)
            ),
        )
        drafter_model = EncoderOnlyFourHeadDrafter(
            diffusion_config, dataset, tokenizer
        ).to(device)
    elif backbone_architecture == 'masked_decoder':
        if not args.diffusion_checkpoint:
            raise ValueError(
                '--diffusion-checkpoint is required for masked_decoder drafters'
            )
        drafter_model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
        drafter_model.load_state_dict(
            torch.load(args.diffusion_checkpoint, map_location=device)
        )
    else:
        raise ValueError(f'unsupported drafter backbone: {backbone_architecture}')
    drafter_model.load_state_dict(drafter_state['model'])
    selector_rank = int(checkpoint_args.get('pair_rank', args.pair_rank))
    selector = PairwisePathSelector(
        drafter_model.n_digit,
        drafter_model.codebook_size,
        drafter_model.n_embd,
        rank=selector_rank,
    ).to(device)
    selector.load_state_dict(drafter_state['selector'])
    drafter_model.eval()
    selector.eval()
    for parameter in drafter_model.parameters():
        parameter.requires_grad_(False)
    for parameter in selector.parameters():
        parameter.requires_grad_(False)

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    teacher_model = None
    if args.teacher_distill_weight > 0 or args.preference_weight > 0:
        teacher_model = copy.deepcopy(ar_model).eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    trainable_parameters = configure_trainable_scope(
        ar_model, args.trainable_scope
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    prefix_counts = [
        int(torch.unique(catalog[:, :length], dim=0).shape[0])
        for length in range(1, ar_model.n_digit + 1)
    ]
    prefix_adaptive_weights = None

    history = []
    ar_model.eval()
    initial_validation = evaluate(
        drafter_model,
        selector,
        val_loader,
        catalog,
        args.proposal_k,
        ar_model=ar_model,
        fusion_alphas=fusion_alphas,
        description='candidate-aware initial validation',
    )
    best_score, initial_alpha = validation_score(
        initial_validation,
        fusion_alphas,
        metric_k=min(10, int(args.proposal_k), int(catalog.shape[0])),
    )
    best_epoch = 0
    no_improve = 0
    checkpoint_path = output_dir / 'best_ar.pt'
    started = time.perf_counter()
    initial_record = {
        'epoch': 0,
        'selected_fusion_alpha': initial_alpha,
        'selection_ndcg@10': best_score,
        'validation': initial_validation,
    }
    history.append(initial_record)
    torch.save(
        {
            'model': ar_model.state_dict(),
            'args': vars(args),
            'epoch': 0,
            'selected_fusion_alpha': initial_alpha,
            'validation': initial_validation,
        },
        checkpoint_path,
    )
    print(json.dumps(initial_record, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        ar_model.train()
        losses = []
        token_losses = []
        listwise_losses = []
        margin_losses = []
        distill_losses = []
        prefix_losses = []
        preference_losses = []
        prefix_component_sums = np.zeros(ar_model.n_digit, dtype=np.float64)
        last_prefix_weights = None
        lambda_nonzero_rates = []
        positive_rates = []
        proposal_positive_rates = []
        margins = []
        recalled_rates = []
        recalled_head_rates = []
        recalled_middle_rates = []
        recalled_tail_rates = []
        for batch in tqdm(train_loader, desc=f'verifier train epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            targets = batch['decoder_labels'].to(device)
            (
                candidates,
                candidate_proposal_scores,
                recalled,
                target_ranks,
            ) = proposal_negatives(
                drafter_model,
                selector,
                batch,
                catalog,
                targets,
                args.num_negatives,
                args.proposal_k,
                negative_strata,
                args.negative_mining,
                ar_model,
                args.candidate_score_chunk_size,
            )
            token_loss = ar_model(batch, return_loss=True).loss
            candidate_token_scores = candidate_token_log_scores(
                ar_model,
                batch,
                candidates,
                args.candidate_score_chunk_size,
            )
            path_scores = candidate_token_scores.sum(dim=-1)
            if args.ranking_score == 'fused':
                ranking_scores = (
                    (1.0 - float(args.training_fusion_alpha))
                    * normalize_scores(candidate_proposal_scores)
                    + float(args.training_fusion_alpha)
                    * normalize_scores(path_scores)
                )
            else:
                ranking_scores = path_scores
            per_example_listwise = F.cross_entropy(
                ranking_scores / args.listwise_temperature,
                torch.zeros(
                    ranking_scores.shape[0], dtype=torch.long, device=device
                ),
                reduction='none',
            )
            example_weights = rank_weights(target_ranks, args)
            if args.require_target_in_proposals:
                example_weights = example_weights * recalled.float()
            listwise_loss = weighted_mean(
                per_example_listwise, example_weights
            )
            positive_margin = (
                ranking_scores[:, 0]
                - ranking_scores[:, 1:].max(dim=1).values
            )
            per_example_margin = F.relu(
                float(args.margin_value) - positive_margin
            )
            margin_loss = weighted_mean(
                per_example_margin, example_weights
            )
            if args.prefix_rank_weight > 0:
                (
                    prefix_loss,
                    prefix_components,
                    prefix_adaptive_weights,
                ) = prefix_pairwise_loss(
                    candidate_token_scores,
                    candidates,
                    example_weights,
                    args.prefix_rank_temperature,
                    prefix_counts,
                    negative_scale=args.prefix_negative_scale,
                    previous_adaptive_weights=prefix_adaptive_weights,
                    adaptive_eta=args.prefix_adaptive_eta,
                )
                prefix_component_sums += prefix_components.detach().cpu().numpy()
                last_prefix_weights = prefix_adaptive_weights.cpu().numpy()
            else:
                prefix_loss = path_scores.new_zeros(())
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_scores = teacher_model.score_candidate_paths(
                        batch,
                        candidates,
                        chunk_size=args.candidate_score_chunk_size,
                    )
                temperature = float(args.teacher_temperature)
                teacher_probs = F.softmax(teacher_scores / temperature, dim=1)
                student_log_probs = F.log_softmax(
                    path_scores / temperature, dim=1
                )
                distill_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction='batchmean',
                ) * temperature**2
            else:
                teacher_scores = None
                distill_loss = path_scores.new_zeros(())
            if args.preference_weight > 0:
                preference_loss, lambda_weights = ndcg_lambda_dpo_loss(
                    path_scores,
                    teacher_scores,
                    candidate_proposal_scores,
                    example_weights,
                    args.preference_beta,
                    args.preference_ndcg_k,
                    args.training_fusion_alpha,
                )
                lambda_nonzero_rates.append(float(
                    lambda_weights.gt(0).float().mean().detach()
                ))
            else:
                preference_loss = path_scores.new_zeros(())
            loss = (
                float(args.token_loss_weight) * token_loss
                + float(args.listwise_weight) * listwise_loss
                + float(args.margin_weight) * margin_loss
                + float(args.teacher_distill_weight) * distill_loss
                + float(args.prefix_rank_weight) * prefix_loss
                + float(args.preference_weight) * preference_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach()))
            token_losses.append(float(token_loss.detach()))
            listwise_losses.append(float(listwise_loss.detach()))
            margin_losses.append(float(margin_loss.detach()))
            distill_losses.append(float(distill_loss.detach()))
            prefix_losses.append(float(prefix_loss.detach()))
            preference_losses.append(float(preference_loss.detach()))
            positive_rates.append(
                float(ranking_scores.argmax(dim=1).eq(0).float().mean().detach())
            )
            proposal_positive_rates.append(
                float(
                    candidate_proposal_scores.argmax(dim=1)
                    .eq(0).float().mean().detach()
                )
            )
            margins.append(
                float(positive_margin.mean().detach())
            )
            recalled_rates.append(float(recalled.float().mean()))
            recalled_head_rates.append(float(target_ranks.le(10).float().mean()))
            recalled_middle_rates.append(float(
                target_ranks.gt(10).logical_and(target_ranks.le(32)).float().mean()
            ))
            recalled_tail_rates.append(float(
                target_ranks.gt(32).logical_and(target_ranks.le(args.proposal_k)).float().mean()
            ))

        ar_model.eval()
        validation = evaluate(
            drafter_model,
            selector,
            val_loader,
            catalog,
            args.proposal_k,
            ar_model=ar_model,
            fusion_alphas=fusion_alphas,
            description=f'candidate-aware validation epoch {epoch}',
        )
        selection_k = min(10, int(args.proposal_k), int(catalog.shape[0]))
        score, selected_alpha = validation_score(
            validation, fusion_alphas, metric_k=selection_k
        )
        epoch_record = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'train_token_loss': float(np.mean(token_losses)),
            'train_listwise_loss': float(np.mean(listwise_losses)),
            'train_margin_loss': float(np.mean(margin_losses)),
            'train_teacher_distill_loss': float(np.mean(distill_losses)),
            'train_prefix_rank_loss': float(np.mean(prefix_losses)),
            'train_preference_loss': float(np.mean(preference_losses)),
            'train_prefix_component_losses': (
                (prefix_component_sums / max(len(losses), 1)).tolist()
            ),
            'train_prefix_adaptive_weights': (
                None if last_prefix_weights is None
                else last_prefix_weights.tolist()
            ),
            'train_lambda_nonzero_rate': (
                None if not lambda_nonzero_rates
                else float(np.mean(lambda_nonzero_rates))
            ),
            'train_positive_top1_rate': float(np.mean(positive_rates)),
            'train_proposal_positive_top1_rate': float(
                np.mean(proposal_positive_rates)
            ),
            'train_positive_margin': float(np.mean(margins)),
            'train_candidate_recall': float(np.mean(recalled_rates)),
            'train_target_rank_1_10': float(np.mean(recalled_head_rates)),
            'train_target_rank_11_32': float(np.mean(recalled_middle_rates)),
            'train_target_rank_33_k': float(np.mean(recalled_tail_rates)),
            'selected_fusion_alpha': selected_alpha,
            'selection_ndcg@10': score,
            'validation': validation,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    'model': ar_model.state_dict(),
                    'args': vars(args),
                    'epoch': epoch,
                    'selected_fusion_alpha': selected_alpha,
                    'validation': validation,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if epoch >= args.min_epochs and no_improve >= args.patience:
                break

    best = torch.load(checkpoint_path, map_location=device)
    ar_model.load_state_dict(best['model'])
    ar_model.eval()
    test = evaluate(
        drafter_model,
        selector,
        test_loader,
        catalog,
        args.proposal_k,
        ar_model=ar_model,
        fusion_alphas=[best['selected_fusion_alpha']],
        description='candidate-aware test',
    )
    report = {
        'protocol': vars(args),
        'catalog_items': int(catalog.shape[0]),
        'best_epoch': best_epoch,
        'best_validation_ndcg@10': best_score,
        'selected_fusion_alpha': best['selected_fusion_alpha'],
        'best_validation': best['validation'],
        'history': history,
        'test': test,
        'elapsed_seconds': time.perf_counter() - started,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(test, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
