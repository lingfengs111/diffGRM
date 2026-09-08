#!/usr/bin/env python
"""Train/evaluate a one-pass OPQ drafter with optional pairwise selection.

The drafter performs exactly one full-mask decoder pass.  Complete items are
ranked over the collision-free legal catalog, then the existing AR model
teacher-forces those candidates as a verifier.  This deliberately removes
iterative reveal order from the proposal stage.
"""

import argparse
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
from genrec.models.DIFF_GRM.parallel_drafter import (
    PairwisePathSelector,
    batched_catalog_unary_scores,
    candidate_tree_diagnostics,
    catalog_unary_scores,
    code_rows,
    linear_curriculum_weight,
)
from genrec.models.DIFF_GRM.encoder_head_drafter import (
    EncoderOnlyFourHeadDrafter,
)
from genrec.models.DIFF_GRM.history_attention_drafter import (
    HistoryAttentionDrafter, score_interest_catalog,
)
from genrec.models.DIFF_GRM.set_drafter import (
    conditional_catalog_scores,
    masked_standardize,
    sample_typed_subset_masks,
)
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2014CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--sid-config', default=None)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument(
        '--diffusion-checkpoint',
        default=None,
        help=(
            'Required only for diffusion_pretrained masked-decoder backbones; '
            'the random encoder_four_head control is trained from scratch.'
        ),
    )
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--history-head', choices=('pooled', 'mlp', 'attention'), default='pooled')
    parser.add_argument('--n-interests', type=int, default=1)
    parser.add_argument('--interest-temperature', type=float, default=1.0)
    parser.add_argument('--validation-only', action='store_true',
                        help='Do not evaluate test; write validation results for experiment screening.')
    parser.add_argument('--retain-candidate-checkpoint', action='store_true',
                        help='Also retain candidate_best.pt selected by proposal recall.')
    parser.add_argument('--skip-tree-diagnostics', action='store_true',
                        help='Skip factorial candidate-tree analysis unrelated to item ranking.')
    parser.add_argument('--dump-selected-ranks', action='store_true',
                        help='Save per-example target ranks for the validation-selected fusion.')
    parser.add_argument(
        '--init-trained-checkpoint',
        default=None,
        help='Optional best.pt from an earlier parallel-drafter run.',
    )
    parser.add_argument(
        '--variant', choices=('unary', 'pairwise', 'triple'), required=True
    )
    parser.add_argument(
        '--backbone-architecture',
        choices=('masked_decoder', 'encoder_four_head'),
        default='masked_decoder',
        help=(
            'Use the DiffGRM masked encoder-decoder or a causal encoder-only '
            'four-head control with the same downstream objectives.'
        ),
    )
    parser.add_argument(
        '--backbone-initialization',
        choices=('diffusion_pretrained', 'random'),
        default='diffusion_pretrained',
        help='Whether the masked backbone loads the denoising checkpoint.',
    )
    parser.add_argument(
        '--encoder-head-n-layer',
        type=int,
        default=4,
        help='History Transformer depth for encoder_four_head.',
    )
    parser.add_argument(
        '--conditioner',
        choices=('diffusion_encoder', 'frozen_ar_encoder'),
        default='diffusion_encoder',
    )
    parser.add_argument('--pair-rank', type=int, default=32)
    parser.add_argument(
        '--allow-pair-rank-expansion', action='store_true',
        help=(
            'Warm-start a wider pairwise selector by copying old factors into '
            'the leading ranks and zero-gating the new ranks.'
        ),
    )
    parser.add_argument(
        '--triple-rank', type=int, default=16,
        help='Low-rank triple residual width; used only by --variant=triple.',
    )
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help=(
            'Optional early-stopping patience in validation evaluations. '
            'The default preserves the historical fixed-epoch protocol.'
        ),
    )
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=0,
        help='Do not early-stop before this many newly trained epochs.',
    )
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--backbone-lr', type=float, default=1e-4)
    parser.add_argument('--selector-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--token-loss-weight', type=float, default=0.1)
    parser.add_argument(
        '--base-anchor-start', type=float, default=0.0,
        help=(
            'Initial weight of unary-only catalog CE in a Domino-style '
            'base-anchored curriculum. Zero preserves the existing objective.'
        ),
    )
    parser.add_argument(
        '--base-anchor-end', type=float, default=0.0,
        help='Final unary-only catalog CE weight after linear annealing.',
    )
    parser.add_argument(
        '--base-anchor-decay-epochs', type=int, default=0,
        help=(
            'Epoch at which the base-anchor weight reaches its final value; '
            'zero uses --epochs.'
        ),
    )
    parser.add_argument(
        '--sampled-catalog-negatives', type=int, default=0,
        help=(
            'Use this many per-example uniform legal-item negatives for '
            'catalog CE during training. Zero computes exact full-catalog CE; '
            'validation and test always rank the full catalog.'
        ),
    )
    parser.add_argument(
        '--mtp-temperature', type=float, default=0.07,
        help='Cosine-logit temperature for the RPG-style mtp_only control.',
    )
    parser.add_argument(
        '--training-objective',
        choices=('catalog_plus_token', 'catalog_only', 'mtp_only'),
        default='catalog_plus_token',
        help=(
            'catalog_plus_token preserves the main item-CE objective; '
            'catalog_only removes auxiliary token CE; mtp_only is the '
            'RPG-style independent multi-token-prediction control.'
        ),
    )
    parser.add_argument(
        '--selection-metric',
        choices=('candidate_recall', 'ndcg10', 'fused_ndcg10'),
        default='candidate_recall',
        help=(
            'Validation checkpoint criterion. RPG-style standalone runs use '
            'ndcg10; proposal models intended for a verifier use candidate_recall.'
        ),
    )
    parser.add_argument(
        '--subset-loss-weight', type=float, default=0.0,
        help=(
            'Weight of typed-set conditional denoising views. Zero preserves '
            'the historical full-mask-only objective.'
        ),
    )
    parser.add_argument('--subset-mask-views', type=int, default=1)
    parser.add_argument('--subset-min-masked', type=int, default=1)
    parser.add_argument(
        '--subset-max-masked', type=int, default=None,
        help='Defaults to n_digit-1, because full-mask is already an anchor view.',
    )
    parser.add_argument(
        '--two-pass-branches', type=int, default=0,
        help=(
            'If positive, evaluate a second parallel denoising pass conditioned '
            'on this many values of the most confident coordinate.'
        ),
    )
    parser.add_argument(
        '--two-pass-first-weights', default='0,0.25,0.5,0.75,1',
        help='Grid for first-pass versus conditional second-pass score fusion.',
    )
    parser.add_argument(
        '--two-pass-branch-chunk', type=int, default=None,
        help='Physical branch chunk; defaults to all branches in one second pass.',
    )
    parser.add_argument(
        '--two-pass-preserve-first', action='store_true',
        help=(
            'Use a self-correcting fallback: items outside the committed route '
            'retain their first-pass score instead of being deleted.'
        ),
    )
    parser.add_argument(
        '--skip-ar-verifier', action='store_true',
        help='Report standalone drafter results without loading/scoring the AR model.',
    )
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument('--fusion-alpha', type=float, default=0.5)
    parser.add_argument(
        '--fusion-alphas',
        default=None,
        help='Validation grid, e.g. 0,0.25,0.5,0.75,1. Defaults to fusion-alpha.',
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


def normalize_scores(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def load_expanded_pairwise_state(selector, source_state):
    """Load a narrower pairwise selector as an exact wider-rank warm start."""
    target_state = selector.state_dict()
    source_rank = int(source_state['left'].shape[-1])
    target_rank = int(target_state['left'].shape[-1])
    unexpected = sorted(set(source_state) - set(target_state))
    if unexpected:
        raise ValueError(f'unexpected selector keys: {unexpected}')
    for key, source in source_state.items():
        target = target_state[key]
        if target.shape == source.shape:
            target.copy_(source)
            continue
        if key in ('left', 'right'):
            if target.shape[:-1] != source.shape[:-1] or target.shape[-1] < source.shape[-1]:
                raise ValueError(
                    f'cannot expand {key} from {source.shape} to {target.shape}'
                )
            target[..., :source.shape[-1]].copy_(source)
            continue
        if key.startswith('gates.') and key.endswith('.weight'):
            if target.shape[1:] != source.shape[1:] or target.shape[0] < source.shape[0]:
                raise ValueError(
                    f'cannot expand {key} from {source.shape} to {target.shape}'
                )
            target.zero_()
            target[:source.shape[0]].copy_(source)
            continue
        if key.startswith('gates.') and key.endswith('.bias'):
            if target.shape[0] < source.shape[0]:
                raise ValueError(
                    f'cannot expand {key} from {source.shape} to {target.shape}'
                )
            target.zero_()
            target[:source.shape[0]].copy_(source)
            continue
        raise ValueError(
            f'unsupported selector shape change for {key}: '
            f'{source.shape} -> {target.shape}'
        )
    # PairwisePathSelector normalizes its low-rank dot product by sqrt(rank).
    # Compensate for the changed denominator so the expanded model is an exact
    # functional warm start before the newly added rank dimensions learn.
    if target_rank != source_rank:
        target_state['pair_scale'].mul_((target_rank / source_rank) ** 0.5)
    selector.load_state_dict(target_state)


def encode_history(model, batch, conditioning_model=None):
    if conditioning_model is None:
        output = model(batch, return_loss=False)
        if getattr(model, 'uses_full_history', False):
            return {'encoder_hidden': output.hidden_states, 'history_mask': output.history_mask}
        return output.hidden_states
    with torch.no_grad():
        return conditioning_model(batch, return_loss=False).hidden_states


def masked_outputs(
    model,
    encoder_hidden,
    decoder_input_ids,
    mask_positions,
    catalog,
    selector=None,
):
    decoded = model.forward_decoder_only(
        {
            'decoder_input_ids': decoder_input_ids,
            'encoder_hidden': encoder_hidden,
            'mask_positions': mask_positions.float(),
        },
        digit=None,
        use_cache=False,
    )
    unary = catalog_unary_scores(decoded.logits, catalog)
    pairwise = None
    scores = unary
    if selector is not None:
        pairwise = selector(decoded.hidden_states, catalog)
        scores = scores + pairwise
    return scores, decoded.logits, decoded.hidden_states, unary, pairwise


def one_pass_outputs(
    model,
    batch,
    catalog,
    selector=None,
    conditioning_model=None,
    encoder_hidden=None,
):
    if encoder_hidden is None:
        encoder_hidden = encode_history(model, batch, conditioning_model)
    decoded = one_pass_decode(model, encoder_hidden)
    if decoded.logits.ndim == 4:
        return score_interest_catalog(model, decoded, catalog, selector)
    unary = catalog_unary_scores(decoded.logits, catalog)
    pairwise = None
    scores = unary
    if selector is not None:
        pairwise = selector(decoded.hidden_states, catalog)
        scores = scores + pairwise
    return scores, decoded.logits, decoded.hidden_states, unary, pairwise


def one_pass_decode(model, encoder_hidden):
    """Predict all SID coordinates without materializing catalog scores."""
    if isinstance(encoder_hidden, dict):
        return model.forward_decoder_only(encoder_hidden)
    batch_size = encoder_hidden.shape[0]
    device = encoder_hidden.device
    return model.forward_decoder_only(
        {
            'decoder_input_ids': torch.zeros(
                batch_size, model.n_digit, dtype=torch.long, device=device
            ),
            'encoder_hidden': encoder_hidden,
            'mask_positions': torch.ones(
                batch_size, model.n_digit, dtype=torch.float32, device=device
            ),
        },
        digit=None,
        use_cache=False,
    )


def sampled_catalog_outputs(
    model,
    encoder_hidden,
    catalog,
    target_rows,
    n_negatives,
    selector=None,
):
    """Score target plus uniform negatives without scanning the full catalog.

    Negatives are sampled with replacement from all non-target rows.  Scaling
    their exponentiated scores by ``(N-1)/K`` gives a Monte Carlo estimate of
    the omitted full-softmax denominator while keeping a fixed training cost.
    """
    n_items = int(catalog.shape[0])
    n_negatives = int(n_negatives)
    if not 0 < n_negatives < n_items:
        raise ValueError(
            'sampled catalog negatives must lie in [1, n_items-1]'
        )
    batch_size = int(target_rows.shape[0])
    negative_rows = torch.randint(
        0,
        n_items - 1,
        (batch_size, n_negatives),
        device=target_rows.device,
    )
    # Bijection from [0,N-2] to every row except this example's target.
    negative_rows = negative_rows + negative_rows.ge(target_rows[:, None])
    sampled_rows = torch.cat([target_rows[:, None], negative_rows], dim=1)
    sampled_codes = catalog[sampled_rows]

    decoded = one_pass_decode(model, encoder_hidden)
    unary_scores = batched_catalog_unary_scores(decoded.logits, sampled_codes)
    scores = unary_scores
    if selector is not None:
        scores = scores + selector(decoded.hidden_states, sampled_codes)

    scores = scores.clone()
    unary_scores = unary_scores.clone()
    scores[:, 1:] = scores[:, 1:] + np.log(
        float(n_items - 1) / float(n_negatives)
    )
    unary_scores[:, 1:] = unary_scores[:, 1:] + np.log(
        float(n_items - 1) / float(n_negatives)
    )
    return scores, decoded.logits, unary_scores


def subset_denoising_loss(
    model,
    encoder_hidden,
    targets,
    target_rows,
    catalog,
    selector,
    n_views,
    min_masked,
    max_masked,
    token_loss_weight,
):
    """Conditional item/token loss on arbitrary typed coordinate subsets."""
    batch_size, n_digit = targets.shape
    masks = sample_typed_subset_masks(
        batch_size,
        n_digit,
        n_views=n_views,
        min_masked=min_masked,
        max_masked=max_masked,
        device=targets.device,
    )
    flat_masks = masks.reshape(-1, n_digit)
    repeated_targets = targets[:, None, :].expand(
        -1, n_views, -1
    ).reshape(-1, n_digit)
    decoder_input = repeated_targets.masked_fill(flat_masks, 0)
    repeated_hidden = encoder_hidden[:, None].expand(
        -1, n_views, *encoder_hidden.shape[1:]
    ).reshape(-1, *encoder_hidden.shape[1:])
    _, logits, hidden, _, structural = masked_outputs(
        model,
        repeated_hidden,
        decoder_input,
        flat_masks,
        catalog,
        selector,
    )
    conditional, _ = conditional_catalog_scores(
        logits,
        catalog,
        repeated_targets,
        flat_masks,
        structural_scores=structural,
    )
    repeated_rows = target_rows[:, None].expand(-1, n_views).reshape(-1)
    item_loss = F.cross_entropy(conditional, repeated_rows)
    token_losses = torch.zeros(
        repeated_targets.shape[0], device=targets.device
    )
    for digit in range(n_digit):
        token_losses = token_losses + F.cross_entropy(
            logits[:, digit], repeated_targets[:, digit], reduction='none'
        ) * flat_masks[:, digit].float()
    token_loss = (
        token_losses / flat_masks.sum(dim=1).clamp_min(1).float()
    ).mean()
    return item_loss + float(token_loss_weight) * token_loss, item_loss, token_loss


@torch.no_grad()
def two_pass_scores(
    model,
    encoder_hidden,
    catalog,
    selector,
    first_scores,
    first_logits,
    branches,
    first_weights,
    branch_chunk=None,
    preserve_first=False,
):
    """One full-mask pass followed by one batched typed-set refinement pass.

    The first pass chooses the lowest-entropy coordinate independently for each
    query.  Its top values form parallel hypotheses.  The second pass observes
    one typed coordinate and denoises all remaining coordinates simultaneously.
    Branch chunking changes physical memory use, not the proposal distribution.
    """
    batch_size, n_digit, codebook_size = first_logits.shape
    branches = min(int(branches), int(codebook_size))
    branch_chunk = branches if branch_chunk is None else int(branch_chunk)
    if branches <= 0 or branch_chunk <= 0:
        raise ValueError('two-pass branch counts must be positive')
    first_log_probs = F.log_softmax(first_logits, dim=-1)
    first_probs = first_log_probs.exp()
    entropy = -(first_probs * first_log_probs).sum(dim=-1)
    reveal_digit = entropy.argmin(dim=1)
    chosen_logits = first_log_probs[
        torch.arange(batch_size, device=first_logits.device), reveal_digit
    ]
    branch_values = chosen_logits.topk(branches, dim=1).indices

    conditional = first_scores.new_full(first_scores.shape, float('-inf'))
    for start in range(0, branches, branch_chunk):
        stop = min(start + branch_chunk, branches)
        width = stop - start
        values = branch_values[:, start:stop]
        flat_hidden = encoder_hidden[:, None].expand(
            -1, width, *encoder_hidden.shape[1:]
        ).reshape(-1, *encoder_hidden.shape[1:])
        flat_input = torch.zeros(
            batch_size * width,
            n_digit,
            dtype=torch.long,
            device=first_logits.device,
        )
        flat_mask = torch.ones_like(flat_input, dtype=torch.bool)
        flat_digits = reveal_digit[:, None].expand(-1, width).reshape(-1)
        flat_values = values.reshape(-1)
        rows = torch.arange(batch_size * width, device=first_logits.device)
        flat_input[rows, flat_digits] = flat_values
        flat_mask[rows, flat_digits] = False
        _, logits, hidden, _, structural = masked_outputs(
            model,
            flat_hidden,
            flat_input,
            flat_mask,
            catalog,
            selector,
        )
        branch_scores, _ = conditional_catalog_scores(
            logits,
            catalog,
            flat_input,
            flat_mask,
            structural_scores=structural,
        )
        branch_scores = branch_scores.reshape(
            batch_size, width, catalog.shape[0]
        ).amax(dim=1)
        conditional = torch.maximum(conditional, branch_scores)

    valid = torch.isfinite(conditional)
    normalized_first = masked_standardize(
        first_scores, torch.ones_like(valid)
    )
    normalized_conditional = masked_standardize(conditional, valid)
    fused = {}
    for weight in first_weights:
        weight = float(weight)
        mixed = (
            weight * normalized_first
            + (1.0 - weight) * normalized_conditional
        )
        if preserve_first:
            # A wrong route must not erase a good first-pass proposal.  This
            # mirrors self-correcting diffusion and makes weight=1 an exact
            # one-pass control under the same evaluation code.
            scores = torch.where(valid, mixed, normalized_first)
        else:
            scores = mixed.masked_fill(~valid, float('-inf'))
        fused[weight] = scores
    return fused, reveal_digit, branch_values, valid


def ranking_metrics(ranked_codes, labels, cutoffs=(5, 10)):
    matches = ranked_codes.eq(labels[:, None, :]).all(dim=-1)
    output = {}
    for cutoff in cutoffs:
        cutoff = min(int(cutoff), ranked_codes.shape[1])
        hits = matches[:, :cutoff].any(dim=1)
        positions = torch.arange(
            ranked_codes.shape[1], device=ranked_codes.device
        )[None]
        first = torch.where(matches, positions, ranked_codes.shape[1]).min(dim=1).values
        output[f'recall@{cutoff}'] = hits.float()
        output[f'ndcg@{cutoff}'] = torch.where(
            hits,
            torch.log2(first.float() + 2.0).reciprocal(),
            torch.zeros_like(first, dtype=torch.float),
        )
    return output


@torch.no_grad()
def evaluate(
    model,
    selector,
    loader,
    catalog,
    proposal_k,
    ar_model=None,
    fusion_alphas=(),
    conditioning_model=None,
    description='evaluation',
    two_pass_branches=0,
    two_pass_first_weights=(),
    two_pass_branch_chunk=None,
    two_pass_preserve_first=False,
    skip_tree_diagnostics=False,
    rank_dump_path=None,
):
    model.eval()
    if selector is not None:
        selector.eval()
    if ar_model is not None:
        ar_model.eval()
    aggregates = {}
    n_examples = 0
    rank_dumps = {'target_codes': [], 'drafter_rank': [], 'ar_rank': [], 'fused_rank': []}
    started = time.perf_counter()
    for batch in tqdm(loader, desc=description):
        labels = batch['labels'].to(catalog.device)
        encoder_hidden = encode_history(model, batch, conditioning_model)
        scores, logits, _, unary_scores, _ = one_pass_outputs(
            model,
            batch,
            catalog,
            selector,
            conditioning_model=conditioning_model,
            encoder_hidden=encoder_hidden,
        )
        keep = min(int(proposal_k), catalog.shape[0])
        proposal_scores, proposal_rows = torch.topk(scores, k=keep, dim=1)
        proposals = catalog[proposal_rows]
        if not skip_tree_diagnostics:
            tree = candidate_tree_diagnostics(proposals, model.codebook_size)
            for name in (
                'independent_nodes', 'fixed_order_nodes', 'best_order_nodes',
                'best_vs_fixed_ratio', 'best_vs_independent_ratio',
            ):
                aggregates.setdefault(f'tree_{name}', []).extend(tree[name].cpu().tolist())
            for order_idx, order in enumerate(tree['orders']):
                tag = ''.join(str(digit) for digit in order)
                aggregates.setdefault(f'tree_best_order_{tag}_rate', []).extend(
                    tree['best_order_index'].eq(order_idx).float().cpu().tolist()
                )
        for name, values in getattr(model, 'last_interest_diagnostics', {}).items():
            aggregates.setdefault(name, []).extend(values.cpu().tolist())
        if rank_dump_path is not None:
            rank_dumps['target_codes'].append(labels.cpu().numpy())
            rank_dumps['drafter_rank'].append(target_ranks(proposals, labels).cpu().numpy())
        metrics = ranking_metrics(proposals, labels, cutoffs=(5, 10, keep))
        for name, values in metrics.items():
            aggregates.setdefault(f'drafter_{name}', []).extend(values.cpu().tolist())
        if selector is not None:
            unary_rows = torch.topk(unary_scores, k=keep, dim=1).indices
            unary_ranked = catalog[unary_rows]
            unary_metrics = ranking_metrics(
                unary_ranked, labels, cutoffs=(5, 10, keep)
            )
            for name, values in unary_metrics.items():
                aggregates.setdefault(f'unary_base_{name}', []).extend(
                    values.cpu().tolist()
                )

        if two_pass_branches:
            second_scores, reveal_digit, branch_values, valid = two_pass_scores(
                model,
                encoder_hidden,
                catalog,
                selector,
                scores,
                logits,
                two_pass_branches,
                two_pass_first_weights,
                branch_chunk=two_pass_branch_chunk,
                preserve_first=two_pass_preserve_first,
            )
            label_at_reveal = labels.gather(1, reveal_digit[:, None]).squeeze(1)
            route_hit = branch_values.eq(label_at_reveal[:, None]).any(dim=1)
            aggregates.setdefault(
                f'two_pass_route_recall@{branch_values.shape[1]}', []
            ).extend(route_hit.float().cpu().tolist())
            aggregates.setdefault('two_pass_legal_fraction', []).extend(
                valid.float().mean(dim=1).cpu().tolist()
            )
            for weight, refined_scores in second_scores.items():
                refined_rows = torch.topk(
                    refined_scores, k=keep, dim=1
                ).indices
                refined = catalog[refined_rows]
                refined_metrics = ranking_metrics(
                    refined, labels, cutoffs=(5, 10, keep)
                )
                tag = f'{float(weight):g}'.replace('.', 'p')
                for name, values in refined_metrics.items():
                    aggregates.setdefault(
                        f'two_pass_w{tag}_{name}', []
                    ).extend(values.cpu().tolist())

        if ar_model is not None:
            ar_scores = ar_model.score_candidate_paths(batch, proposals)
            ar_order = ar_scores.argsort(dim=1, descending=True)
            ar_ranked = proposals.gather(
                1, ar_order.unsqueeze(-1).expand_as(proposals)
            )
            ar_metrics = ranking_metrics(ar_ranked, labels)
            if rank_dump_path is not None:
                rank_dumps['ar_rank'].append(target_ranks(ar_ranked, labels).cpu().numpy())
            for name, values in ar_metrics.items():
                aggregates.setdefault(f'ar_verified_{name}', []).extend(
                    values.cpu().tolist()
                )

            for alpha in fusion_alphas:
                fused = (
                    (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                    + float(alpha) * normalize_scores(ar_scores)
                )
                fused_order = fused.argsort(dim=1, descending=True)
                fused_ranked = proposals.gather(
                    1, fused_order.unsqueeze(-1).expand_as(proposals)
                )
                fused_metrics = ranking_metrics(fused_ranked, labels)
                if rank_dump_path is not None:
                    if len(fusion_alphas) != 1:
                        raise ValueError('rank dumps require one validation-selected alpha')
                    rank_dumps['fused_rank'].append(target_ranks(fused_ranked, labels).cpu().numpy())
                tag = f'{float(alpha):g}'.replace('.', 'p')
                for name, values in fused_metrics.items():
                    aggregates.setdefault(f'fused_a{tag}_{name}', []).extend(
                        values.cpu().tolist()
                    )
        n_examples += labels.shape[0]

    elapsed = time.perf_counter() - started
    if rank_dump_path is not None:
        np.savez_compressed(rank_dump_path, **{
            name: np.concatenate(values) for name, values in rank_dumps.items() if values
        })
    result = {name: float(np.mean(values)) for name, values in aggregates.items()}
    result.update(
        n_examples=n_examples,
        elapsed_seconds=elapsed,
        milliseconds_per_example=1000.0 * elapsed / max(n_examples, 1),
        backbone_passes=(2 if two_pass_branches else 1),
        parallel_second_pass_branches=int(two_pass_branches),
    )
    return result


def target_ranks(codes, labels):
    """One-based candidate rank, or K+1 for a target missing from the pool."""
    matches = codes.eq(labels[:, None]).all(-1)
    positions = torch.arange(1, codes.shape[1] + 1, device=codes.device)
    return torch.where(matches, positions[None], codes.shape[1] + 1).min(1).values


def select_fusion(validation, alphas):
    """Use only validation NDCG@10; deterministic ties follow grid order."""
    values = {
        float(a): validation[f"fused_a{float(a):g}_ndcg@10".replace('.', 'p')]
        for a in alphas
    }
    alpha = max(values, key=values.get)
    return alpha, values[alpha]


def main():
    args = parse_args()
    if args.n_interests < 1 or args.interest_temperature <= 0:
        raise ValueError('interest count and temperature must be positive')
    if args.n_interests > 1 and args.history_head != 'attention':
        raise ValueError('multiple interests require --history-head attention')
    if args.history_head != 'pooled' and args.backbone_architecture != 'encoder_four_head':
        raise ValueError('history heads require encoder_four_head')
    if args.n_interests > 1 and (args.sampled_catalog_negatives or args.training_objective == 'mtp_only'):
        raise ValueError('multi-interest heads currently require full-catalog item training')
    if args.selection_metric == 'fused_ndcg10' and args.skip_ar_verifier:
        raise ValueError('fused checkpoint selection requires the AR verifier')
    if not 0.0 <= args.fusion_alpha <= 1.0:
        raise ValueError('--fusion-alpha must be in [0,1]')
    fusion_alphas = (
        [float(value) for value in args.fusion_alphas.split(',')]
        if args.fusion_alphas
        else [float(args.fusion_alpha)]
    )
    if not fusion_alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in fusion_alphas):
        raise ValueError('--fusion-alphas must contain values in [0,1]')
    two_pass_first_weights = [
        float(value) for value in args.two_pass_first_weights.split(',')
    ]
    if any(weight < 0.0 or weight > 1.0 for weight in two_pass_first_weights):
        raise ValueError('--two-pass-first-weights must lie in [0,1]')
    if args.subset_loss_weight < 0.0 or args.subset_mask_views <= 0:
        raise ValueError('subset loss weight/views must be non-negative/positive')
    if not 0.0 <= args.base_anchor_start <= 1.0:
        raise ValueError('--base-anchor-start must be in [0,1]')
    if not 0.0 <= args.base_anchor_end <= 1.0:
        raise ValueError('--base-anchor-end must be in [0,1]')
    if args.base_anchor_decay_epochs < 0:
        raise ValueError('--base-anchor-decay-epochs must be non-negative')
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
        {'train_batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size},
    )
    ar_config = make_config(
        'AR_GRM', args.dataset,
        common_files + [args.ar_config], accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    device = torch.device(diffusion_config['device'])
    dataset = get_dataset(args.dataset)(diffusion_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    raw_splits = dataset.split()
    # Limiting before tokenization keeps smoke tests genuinely small while the
    # full-data path (all maxima are None) remains byte-for-byte unchanged.
    raw_splits['train'] = limit_dataset(
        raw_splits['train'], args.max_train_examples
    )
    raw_splits['val'] = limit_dataset(raw_splits['val'], args.max_val_examples)
    raw_splits['test'] = limit_dataset(
        raw_splits['test'], args.max_test_examples
    )
    tokenized = tokenizer.tokenize(raw_splits)
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('parallel item training requires collision-free catalog SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    train_data = limit_dataset(tokenized['train'], args.max_train_examples)
    val_data = limit_dataset(tokenized['val'], args.max_val_examples)
    test_data = limit_dataset(tokenized['test'], args.max_test_examples)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=tokenizer.collate_fn['train'],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )

    diffusion_config['encoder_head_n_layer'] = args.encoder_head_n_layer
    diffusion_config['encoder_head_normalize_logits'] = (
        args.training_objective == 'mtp_only'
    )
    diffusion_config['encoder_head_logit_temperature'] = args.mtp_temperature
    diffusion_config.update(history_head=args.history_head, n_interests=args.n_interests,
                            interest_temperature=args.interest_temperature)
    if args.backbone_architecture == 'encoder_four_head':
        if args.backbone_initialization != 'random':
            raise ValueError(
                'encoder_four_head has no diffusion checkpoint; use '
                '--backbone-initialization random'
            )
        if args.conditioner != 'diffusion_encoder':
            raise ValueError('encoder_four_head does not use a separate conditioner')
        if args.subset_loss_weight or args.two_pass_branches:
            raise ValueError(
                'encoder_four_head is a strict one-pass control and does not '
                'support subset/two-pass denoising'
            )
        drafter_class = EncoderOnlyFourHeadDrafter if args.history_head == 'pooled' else HistoryAttentionDrafter
        model = drafter_class(
            diffusion_config, dataset, tokenizer
        ).to(device)
    else:
        model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
        if args.backbone_initialization == 'diffusion_pretrained':
            if not args.diffusion_checkpoint:
                raise ValueError(
                    '--diffusion-checkpoint is required for '
                    'diffusion_pretrained masked_decoder'
                )
            model.load_state_dict(
                torch.load(args.diffusion_checkpoint, map_location=device)
            )
        else:
            print('[INIT] masked decoder is trained from random initialization')
    # Make selector initialization identical across architecture arms.  The
    # train-loader order is controlled by its own generator above.
    torch.manual_seed(args.seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1)
    selector = None
    if args.variant in ('pairwise', 'triple'):
        selector = PairwisePathSelector(
            model.n_digit,
            model.codebook_size,
            model.n_embd,
            rank=args.pair_rank,
            triple_rank=args.triple_rank if args.variant == 'triple' else 0,
        ).to(device)
    if args.init_trained_checkpoint:
        initialized = torch.load(args.init_trained_checkpoint, map_location=device)
        model.load_state_dict(initialized['model'])
        if selector is not None:
            if initialized.get('selector') is None:
                raise ValueError('pairwise variant requires selector weights')
            if args.variant == 'triple':
                incompatible = selector.load_state_dict(
                    initialized['selector'], strict=False
                )
                unexpected = list(incompatible.unexpected_keys)
                missing = [
                    key for key in incompatible.missing_keys
                    if not key.startswith('triple_')
                ]
                if unexpected or missing:
                    raise ValueError(
                        'Triple warm start is incompatible with the selector: '
                        f'missing={missing}, unexpected={unexpected}'
                    )
            elif args.allow_pair_rank_expansion:
                load_expanded_pairwise_state(selector, initialized['selector'])
            else:
                selector.load_state_dict(initialized['selector'])

    ar_model = None
    if args.conditioner == 'frozen_ar_encoder' or args.selection_metric == 'fused_ndcg10':
        # Loading an evaluation-only verifier must not change training dropout
        # or initialization RNG streams across checkpoint-selection controls.
        with torch.random.fork_rng():
            ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
        ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
        ar_model.eval()
        for parameter in ar_model.parameters():
            parameter.requires_grad_(False)
    conditioning_model = ar_model if args.conditioner == 'frozen_ar_encoder' else None

    subset_max_masked = (
        model.n_digit - 1
        if args.subset_max_masked is None else args.subset_max_masked
    )
    if not 1 <= args.subset_min_masked <= subset_max_masked <= model.n_digit:
        raise ValueError('invalid typed-subset mask width range')
    if args.training_objective == 'mtp_only':
        if args.backbone_architecture != 'encoder_four_head':
            raise ValueError('mtp_only currently targets the matched encoder-four-head control')
        if args.mtp_temperature <= 0:
            raise ValueError('mtp_temperature must be positive')
        if args.variant != 'unary':
            raise ValueError('mtp_only is an independent-coordinate unary control')
        if args.subset_loss_weight:
            raise ValueError('mtp_only does not use conditional subset item losses')
        if args.sampled_catalog_negatives:
            raise ValueError('mtp_only has no catalog CE to sample')
        if args.base_anchor_start or args.base_anchor_end:
            raise ValueError('mtp_only has no catalog base objective to anchor')
    if args.sampled_catalog_negatives < 0:
        raise ValueError('sampled catalog negatives must be non-negative')
    if args.sampled_catalog_negatives >= catalog.shape[0]:
        raise ValueError(
            'sampled catalog negatives must be smaller than the catalog'
        )
    if args.sampled_catalog_negatives and args.subset_loss_weight:
        raise ValueError(
            'sampled main-catalog CE with full-catalog subset loss is not a '
            'clean fixed-budget control'
        )
    if (args.base_anchor_start or args.base_anchor_end) and selector is None:
        raise ValueError('base anchoring requires a pairwise/triple selector')

    base_anchor_decay_epochs = (
        args.epochs
        if args.base_anchor_decay_epochs == 0
        else args.base_anchor_decay_epochs
    )
    if (args.base_anchor_start or args.base_anchor_end) and args.epochs < 1:
        raise ValueError('base anchoring requires at least one training epoch')

    parameter_groups = [
        {
            'params': model.parameters(),
            'lr': args.backbone_lr,
            'weight_decay': args.weight_decay,
        }
    ]
    if selector is not None:
        parameter_groups.append(
            {
                'params': selector.parameters(),
                'lr': args.selector_lr,
                'weight_decay': args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    history = []
    checkpoint_path = output_dir / 'best.pt'
    candidate_checkpoint_path = output_dir / 'candidate_best.pt'
    selection_eval_kwargs = {
        'ar_model': ar_model if args.selection_metric == 'fused_ndcg10' else None,
        'fusion_alphas': fusion_alphas if args.selection_metric == 'fused_ndcg10' else (),
        'skip_tree_diagnostics': args.skip_tree_diagnostics,
    }

    initial = evaluate(
        model, selector, val_loader, catalog, args.proposal_k,
        conditioning_model=conditioning_model,
        description='initial validation',
        **selection_eval_kwargs,
    )
    if args.selection_metric == 'fused_ndcg10':
        initial['selected_fusion_alpha'], initial['selection_fused_ndcg@10'] = select_fusion(initial, fusion_alphas)
    initial['epoch'] = 0
    history.append(initial)
    print(json.dumps(initial, sort_keys=True), flush=True)

    selection_key = 'selection_fused_ndcg@10' if args.selection_metric == 'fused_ndcg10' else (
        'drafter_ndcg@10'
        if args.selection_metric == 'ndcg10'
        else f'drafter_recall@{min(args.proposal_k, catalog.shape[0])}'
    )
    best_selection_value = initial[selection_key]
    candidate_key = f'drafter_recall@{min(args.proposal_k, catalog.shape[0])}'
    best_candidate_value = initial[candidate_key]
    no_improve = 0
    # Treat the initialization as a real candidate.  This is essential for
    # continuation runs: a worse first resumed epoch must not overwrite the
    # earlier best checkpoint.
    torch.save(
        {
            'model': model.state_dict(),
            'selector': None if selector is None else selector.state_dict(),
            'args': vars(args),
            'validation': initial,
            'epoch': 0,
        },
        checkpoint_path,
    )
    if args.retain_candidate_checkpoint:
        torch.save({'model': model.state_dict(),
                    'selector': None if selector is None else selector.state_dict(),
                    'args': vars(args), 'validation': initial, 'epoch': 0}, candidate_checkpoint_path)

    if args.epochs == 0 and not args.init_trained_checkpoint:
        raise ValueError('--epochs=0 requires --init-trained-checkpoint')

    for epoch in range(1, args.epochs + 1):
        model.train()
        if selector is not None:
            selector.train()
        if conditioning_model is not None:
            conditioning_model.eval()
        losses = []
        item_losses = []
        item_objective_losses = []
        base_item_losses = []
        token_losses = []
        subset_losses = []
        subset_item_losses = []
        subset_token_losses = []
        base_anchor_weight = linear_curriculum_weight(
            epoch,
            args.base_anchor_start,
            args.base_anchor_end,
            max(base_anchor_decay_epochs, 1),
        )
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            targets = batch['decoder_labels'].to(device)
            encoder_hidden = encode_history(model, batch, conditioning_model)
            target_rows = None
            item_loss = None
            item_objective = None
            base_item_loss = None
            if args.training_objective == 'mtp_only':
                logits = one_pass_decode(model, encoder_hidden).logits
            else:
                target_rows = code_rows(
                    targets,
                    catalog,
                    model.codebook_size,
                    identity_start_digit=int(
                        diffusion_config.get('item_identity_start_digit', 0)
                    ),
                )
                if args.sampled_catalog_negatives:
                    scores, logits, unary_scores = sampled_catalog_outputs(
                        model,
                        encoder_hidden,
                        catalog,
                        target_rows,
                        args.sampled_catalog_negatives,
                        selector=selector,
                    )
                    item_loss = F.cross_entropy(
                        scores,
                        torch.zeros_like(target_rows),
                    )
                    if selector is not None:
                        base_item_loss = F.cross_entropy(
                            unary_scores,
                            torch.zeros_like(target_rows),
                        )
                else:
                    scores, logits, _, unary_scores, _ = one_pass_outputs(
                        model,
                        batch,
                        catalog,
                        selector,
                        conditioning_model=conditioning_model,
                        encoder_hidden=encoder_hidden,
                    )
                    item_loss = F.cross_entropy(scores, target_rows)
                    if selector is not None:
                        base_item_loss = F.cross_entropy(
                            unary_scores, target_rows
                        )
                item_objective = item_loss
                if base_item_loss is not None and base_anchor_weight:
                    item_objective = (
                        base_anchor_weight * base_item_loss
                        + (1.0 - base_anchor_weight) * item_loss
                    )
            token_loss = torch.stack(
                [
                    F.cross_entropy(logits[:, digit], targets[:, digit])
                    for digit in range(model.n_digit)
                ]
            ).mean()
            if args.training_objective == 'mtp_only':
                loss = token_loss
            elif args.training_objective == 'catalog_only':
                loss = item_objective
            else:
                loss = (
                    item_objective
                    + float(args.token_loss_weight) * token_loss
                )
            subset_loss = None
            subset_item_loss = None
            subset_token_loss = None
            if args.subset_loss_weight:
                (
                    subset_loss,
                    subset_item_loss,
                    subset_token_loss,
                ) = subset_denoising_loss(
                    model,
                    encoder_hidden,
                    targets,
                    target_rows,
                    catalog,
                    selector,
                    args.subset_mask_views,
                    args.subset_min_masked,
                    subset_max_masked,
                    args.token_loss_weight,
                )
                loss = loss + float(args.subset_loss_weight) * subset_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if selector is not None:
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if item_loss is not None:
                item_losses.append(float(item_loss.detach()))
                item_objective_losses.append(float(item_objective.detach()))
            if base_item_loss is not None:
                base_item_losses.append(float(base_item_loss.detach()))
            token_losses.append(float(token_loss.detach()))
            if subset_loss is not None:
                subset_losses.append(float(subset_loss.detach()))
                subset_item_losses.append(float(subset_item_loss.detach()))
                subset_token_losses.append(float(subset_token_loss.detach()))

        validation = evaluate(
            model, selector, val_loader, catalog, args.proposal_k,
            conditioning_model=conditioning_model,
            description=f'validation epoch {epoch}',
            **selection_eval_kwargs,
        )
        if args.selection_metric == 'fused_ndcg10':
            validation['selected_fusion_alpha'], validation['selection_fused_ndcg@10'] = select_fusion(validation, fusion_alphas)
        validation.update(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            train_token_loss=float(np.mean(token_losses)),
            train_catalog_candidates=(
                int(args.sampled_catalog_negatives) + 1
                if args.sampled_catalog_negatives else int(catalog.shape[0])
            ),
            train_base_anchor_weight=float(base_anchor_weight),
        )
        if item_losses:
            validation['train_item_loss'] = float(np.mean(item_losses))
            validation['train_item_objective_loss'] = float(
                np.mean(item_objective_losses)
            )
        if base_item_losses:
            validation['train_base_item_loss'] = float(
                np.mean(base_item_losses)
            )
        if subset_losses:
            validation.update(
                train_subset_loss=float(np.mean(subset_losses)),
                train_subset_item_loss=float(np.mean(subset_item_losses)),
                train_subset_token_loss=float(np.mean(subset_token_losses)),
            )
        history.append(validation)
        # Flush machine-readable progress after each epoch for unattended runs.
        progress_tmp = output_dir / 'progress.json.tmp'
        progress_tmp.write_text(json.dumps({'protocol': vars(args), 'history': history}, indent=2))
        progress_tmp.replace(output_dir / 'progress.json')
        print(json.dumps(validation, sort_keys=True), flush=True)
        if args.retain_candidate_checkpoint and validation[candidate_key] > best_candidate_value:
            best_candidate_value = validation[candidate_key]
            torch.save({'model': model.state_dict(),
                        'selector': None if selector is None else selector.state_dict(),
                        'args': vars(args), 'validation': validation, 'epoch': epoch}, candidate_checkpoint_path)
        if validation[selection_key] > best_selection_value:
            best_selection_value = validation[selection_key]
            no_improve = 0
            torch.save(
                {
                    'model': model.state_dict(),
                    'selector': None if selector is None else selector.state_dict(),
                    'args': vars(args),
                    'validation': validation,
                    'epoch': epoch,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if (
                args.patience is not None
                and epoch >= args.min_epochs
                and no_improve >= args.patience
            ):
                print(
                    f'[EARLY STOP] no {selection_key} improvement for '
                    f'{no_improve} epochs; best={best_selection_value:.8f}',
                    flush=True,
                )
                break

    if args.epochs == 0:
        best = {
            'model': model.state_dict(),
            'selector': None if selector is None else selector.state_dict(),
            'args': vars(args),
            'validation': initial,
        }
    else:
        best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best['model'])
    if selector is not None:
        selector.load_state_dict(best['selector'])
    if ar_model is None and not args.skip_ar_verifier:
        ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
        ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
        ar_model.eval()
        for parameter in ar_model.parameters():
            parameter.requires_grad_(False)
    conditioning_model = (
        ar_model if args.conditioner == 'frozen_ar_encoder' else None
    )
    validation_with_verifier = evaluate(
        model,
        selector,
        val_loader,
        catalog,
        args.proposal_k,
        ar_model=None if args.skip_ar_verifier else ar_model,
        fusion_alphas=() if args.skip_ar_verifier else fusion_alphas,
        conditioning_model=conditioning_model,
        description='validation fusion selection',
        two_pass_branches=args.two_pass_branches,
        two_pass_first_weights=two_pass_first_weights,
        two_pass_branch_chunk=args.two_pass_branch_chunk,
        two_pass_preserve_first=args.two_pass_preserve_first,
        skip_tree_diagnostics=args.skip_tree_diagnostics,
    )
    alpha_scores = {}
    if not args.skip_ar_verifier:
        for alpha in fusion_alphas:
            tag = f'{float(alpha):g}'.replace('.', 'p')
            alpha_scores[float(alpha)] = validation_with_verifier[
                f'fused_a{tag}_ndcg@10'
            ]
    selected_alpha = (
        max(alpha_scores, key=alpha_scores.get) if alpha_scores else None
    )
    if args.dump_selected_ranks:
        evaluate(model, selector, val_loader, catalog, args.proposal_k,
                 ar_model=ar_model, fusion_alphas=[selected_alpha] if selected_alpha is not None else (),
                 conditioning_model=conditioning_model, description='selected validation rank dump',
                 skip_tree_diagnostics=True, rank_dump_path=output_dir / 'validation_ranks.npz')
    test = None if args.validation_only else evaluate(
        model,
        selector,
        test_loader,
        catalog,
        args.proposal_k,
        ar_model=None if args.skip_ar_verifier else ar_model,
        fusion_alphas=() if args.skip_ar_verifier else (
            [selected_alpha] if args.dump_selected_ranks else fusion_alphas
        ),
        conditioning_model=conditioning_model,
        description='test proposal + AR verification',
        two_pass_branches=args.two_pass_branches,
        two_pass_first_weights=two_pass_first_weights,
        two_pass_branch_chunk=args.two_pass_branch_chunk,
        two_pass_preserve_first=args.two_pass_preserve_first,
        skip_tree_diagnostics=args.skip_tree_diagnostics,
        rank_dump_path=(output_dir / 'test_ranks.npz') if args.dump_selected_ranks else None,
    )
    report = {
        'variant': args.variant,
        'backbone_architecture': args.backbone_architecture,
        'backbone_initialization': args.backbone_initialization,
        'backbone_parameters': sum(
            parameter.numel() for parameter in model.parameters()
        ),
        'protocol': vars(args),
        'catalog_items': int(catalog.shape[0]),
        'selector_parameters': 0 if selector is None else sum(
            parameter.numel() for parameter in selector.parameters()
        ),
        'best_validation': best['validation'],
        'validation_with_verifier': validation_with_verifier,
        'selected_fusion_alpha': selected_alpha,
        'history': history,
        'candidate_selection_validation': max(history, key=lambda row: row[candidate_key]),
        'test': test,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(test, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
