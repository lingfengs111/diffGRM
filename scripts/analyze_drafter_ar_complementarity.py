#!/usr/bin/env python
"""Full-test behavioral diagnosis for a one-pass drafter and AR verifier.

The script never trains or mutates a checkpoint.  It reconstructs the exact
one-pass evaluation protocol, verifies aggregate metric parity against the
source ``result.json``, and then saves enough per-example state to study:

* proposal recall versus reranking;
* drafter-only, AR-only, both-correct, and neither-correct examples;
* score/rank migration under validation-selected fusion;
* training exposure, one-hop transition frequency, repetition, history
  complexity, text similarity, and SID geometry;
* optionally, overlap with standalone constrained AR beam generation.

The compressed NPZ is intentionally the source artifact.  New analyses can be
computed from it without running either neural model again.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
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

from genrec.diagnostics import catalog_codes, stable_target_ranks
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.encoder_head_drafter import (
    EncoderOnlyFourHeadDrafter,
)
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.parallel_drafter import (
    PairwisePathSelector,
    code_rows,
)
from genrec.utils import get_dataset
from scripts.train_parallel_opq_drafter import (
    encode_history,
    make_config,
    normalize_scores,
    one_pass_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument(
        '--ar-checkpoint', default=None,
        help='Defaults to the AR checkpoint recorded by the drafter run.',
    )
    parser.add_argument(
        '--expected-result', default=None,
        help='Source result.json used for aggregate parity and alpha inference.',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--proposal-k', type=int, default=None)
    parser.add_argument('--fusion-alpha', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--candidate-score-chunk-size', type=int, default=24)
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--include-standalone-ar', action='store_true')
    parser.add_argument(
        '--save-representations', action='store_true',
        help='Retain pooled drafter/AR history and drafter coordinate states.',
    )
    parser.add_argument('--standalone-top-k', type=int, default=10)
    parser.add_argument(
        '--no-candidate-matrix', action='store_true',
        help='Do not retain K candidate IDs/scores in examples.npz.',
    )
    parser.add_argument('--parity-atol', type=float, default=5e-7)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def resolve_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def alpha_tag(alpha):
    return f'{float(alpha):g}'.replace('.', 'p')


def rank_from_ordered_rows(ordered_rows, target_rows):
    matches = ordered_rows.eq(target_rows[:, None])
    present = matches.any(dim=1)
    rank = matches.float().argmax(dim=1).long() + 1
    return torch.where(present, rank, torch.zeros_like(rank))


def normalized_entropy(scores):
    if scores.shape[1] <= 1:
        return torch.zeros(scores.shape[0], device=scores.device)
    log_probs = F.log_softmax(scores.float(), dim=1)
    return -(log_probs.exp() * log_probs).sum(dim=1) / math.log(scores.shape[1])


def wrong_candidate_margin(scores, candidate_rows, target_rows, target_scores):
    wrong = candidate_rows.ne(target_rows[:, None])
    best_wrong = scores.masked_fill(~wrong, float('-inf')).max(dim=1).values
    return target_scores - best_wrong


def target_counterfactual_rank(candidate_scores, target_scores):
    """Rank target against a candidate list, whether or not it was proposed."""
    return 1 + candidate_scores.gt(target_scores[:, None]).sum(dim=1)


def topk_overlap(left_rows, right_rows, cutoff):
    left = left_rows[:, :cutoff]
    right = right_rows[:, :cutoff]
    return left[:, :, None].eq(right[:, None, :]).any(dim=2).sum(dim=1)


def set_overlap(left_rows, right_rows):
    return left_rows[:, :, None].eq(right_rows[:, None, :]).any(dim=2).sum(dim=1)


def ndcg_from_rank(rank, cutoff):
    rank = np.asarray(rank)
    hit = (rank > 0) & (rank <= int(cutoff))
    values = np.zeros(rank.shape, dtype=np.float64)
    values[hit] = 1.0 / np.log2(rank[hit].astype(np.float64) + 1.0)
    return values


def rank_metrics(rank):
    result = {}
    rank = np.asarray(rank)
    for cutoff in (5, 10):
        hit = (rank > 0) & (rank <= cutoff)
        result[f'recall@{cutoff}'] = float(hit.mean())
        result[f'ndcg@{cutoff}'] = float(ndcg_from_rank(rank, cutoff).mean())
    return result


def append_tensor(storage, name, value, dtype=None):
    array = value.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    storage.setdefault(name, []).append(array)


def concatenate_storage(storage):
    return {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in storage.items()
    }


def load_text_embeddings(dataset, tokenizer, config):
    dimension = int(config.get('sent_emb_pca', 0) or config['sent_emb_dim'])
    suffix = (
        f'_pca{dimension}.sent_emb'
        if int(config.get('sent_emb_pca', 0)) > 0
        else f'_raw_d{dimension}.sent_emb'
    )
    path = (
        Path(dataset.cache_dir)
        / 'processed'
        / f'{tokenizer._embedding_cache_basename()}{suffix}'
    )
    if not path.is_file():
        raise FileNotFoundError(f'text embedding cache not found: {path}')
    embeddings = np.fromfile(path, dtype=np.float32).reshape(-1, dimension)
    if embeddings.shape[0] != dataset.n_items - 1:
        raise ValueError(
            f'text embedding rows {embeddings.shape[0]} != catalog '
            f'{dataset.n_items - 1}'
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings.astype(np.float32, copy=False), path


def build_example_attributes(raw_train, raw_test, dataset, catalog, embeddings, max_history):
    """Build model-independent, row-aligned behavioral attributes."""
    item2id = dataset.item2id
    train_target_count = np.zeros(dataset.n_items, dtype=np.int64)
    transition_count = Counter()
    for sequence in raw_train['item_seq']:
        if not sequence:
            continue
        target = item2id[sequence[-1]]
        train_target_count[target] += 1
        if len(sequence) >= 2:
            transition_count[(item2id[sequence[-2]], target)] += 1

    raw = raw_test
    n_examples = len(raw)
    history_ids = np.zeros((n_examples, max_history), dtype=np.int32)
    history_length_full = np.empty(n_examples, dtype=np.int16)
    history_length = np.empty(n_examples, dtype=np.int16)
    history_unique = np.empty(n_examples, dtype=np.int16)
    user_id = np.empty(n_examples, dtype=np.int32)
    target_item_id = np.empty(n_examples, dtype=np.int32)
    last_item_id = np.empty(n_examples, dtype=np.int32)
    repeat_target = np.empty(n_examples, dtype=np.bool_)
    target_train_count = np.empty(n_examples, dtype=np.int32)
    last_transition_count = np.empty(n_examples, dtype=np.int32)

    users = raw['user']
    sequences = raw['item_seq']
    for index, (user, sequence) in enumerate(zip(users, sequences)):
        target = item2id[sequence[-1]]
        full_history = [item2id[item] for item in sequence[:-1]]
        observed = full_history[-max_history:]
        length = len(observed)
        if length:
            history_ids[index, :length] = observed
        history_length_full[index] = len(full_history)
        history_length[index] = length
        history_unique[index] = len(set(observed))
        user_id[index] = dataset.user2id[user]
        target_item_id[index] = target
        last = observed[-1] if observed else 0
        last_item_id[index] = last
        repeat_target[index] = target in observed
        target_train_count[index] = train_target_count[target]
        last_transition_count[index] = transition_count[(last, target)]

    attributes = {
        'row_index': np.arange(n_examples, dtype=np.int32),
        'user_id': user_id,
        'target_item_id': target_item_id,
        'last_item_id': last_item_id,
        'target_train_count': target_train_count,
        'last_transition_count': last_transition_count,
        'repeat_target': repeat_target,
        'history_length_full': history_length_full,
        'history_length': history_length,
        'history_unique': history_unique,
        'history_unique_ratio': history_unique.astype(np.float32)
        / np.maximum(history_length, 1),
    }

    # Vectorize text and SID relationships in bounded-memory chunks.
    text_last_cosine = np.full(n_examples, np.nan, dtype=np.float32)
    text_max_cosine = np.full(n_examples, np.nan, dtype=np.float32)
    sid_last_hamming = np.full(n_examples, -1, dtype=np.int8)
    sid_min_hamming = np.full(n_examples, -1, dtype=np.int8)
    sid_last_match_count = np.full(n_examples, -1, dtype=np.int8)
    sid_max_match_count = np.full(n_examples, -1, dtype=np.int8)
    for start in tqdm(
        range(0, n_examples, 2048), desc='behavioral text/SID attributes'
    ):
        end = min(start + 2048, n_examples)
        hids = history_ids[start:end]
        tids = target_item_id[start:end]
        valid = hids > 0
        safe_rows = np.maximum(hids - 1, 0)

        target_vectors = embeddings[tids - 1]
        history_vectors = embeddings[safe_rows]
        cosine = np.einsum('bd,bld->bl', target_vectors, history_vectors)
        cosine[~valid] = -np.inf
        maximum = cosine.max(axis=1)
        maximum[~valid.any(axis=1)] = np.nan
        text_max_cosine[start:end] = maximum
        lengths = history_length[start:end]
        has_history = lengths > 0
        rows = np.arange(end - start)
        last_positions = np.maximum(lengths - 1, 0)
        last_cosine = cosine[rows, last_positions]
        last_cosine[~has_history] = np.nan
        text_last_cosine[start:end] = last_cosine

        target_codes = catalog[tids - 1]
        history_codes = catalog[safe_rows]
        matches = (history_codes == target_codes[:, None, :]).sum(axis=2)
        matches[~valid] = -1
        maximum_matches = matches.max(axis=1)
        maximum_matches[~valid.any(axis=1)] = -1
        sid_max_match_count[start:end] = maximum_matches.astype(np.int8)
        sid_min_hamming[start:end] = np.where(
            maximum_matches >= 0, catalog.shape[1] - maximum_matches, -1
        ).astype(np.int8)
        last_matches = matches[rows, last_positions]
        last_matches[~has_history] = -1
        sid_last_match_count[start:end] = last_matches.astype(np.int8)
        sid_last_hamming[start:end] = np.where(
            last_matches >= 0, catalog.shape[1] - last_matches, -1
        ).astype(np.int8)

    attributes.update(
        text_last_cosine=text_last_cosine,
        text_max_cosine=text_max_cosine,
        sid_last_hamming=sid_last_hamming,
        sid_min_hamming=sid_min_hamming,
        sid_last_match_count=sid_last_match_count,
        sid_max_match_count=sid_max_match_count,
    )
    return attributes, history_ids


def instantiate_models(checkpoint, run_args, diffusion_config, ar_config, dataset, tokenizer, device):
    architecture = run_args.get('backbone_architecture', 'masked_decoder')
    diffusion_config['encoder_head_n_layer'] = int(
        run_args.get('encoder_head_n_layer', 4)
    )
    diffusion_config['encoder_head_normalize_logits'] = (
        run_args.get('training_objective', 'catalog_plus_token') == 'mtp_only'
    )
    diffusion_config['encoder_head_logit_temperature'] = float(
        run_args.get('mtp_temperature', 0.07)
    )
    if architecture == 'encoder_four_head':
        model = EncoderOnlyFourHeadDrafter(
            diffusion_config, dataset, tokenizer
        ).to(device)
    elif architecture == 'masked_decoder':
        model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    else:
        raise ValueError(f'unsupported drafter architecture: {architecture}')
    model.load_state_dict(checkpoint['model'])

    variant = run_args.get('variant', 'pairwise')
    selector = None
    if variant in ('pairwise', 'triple'):
        selector = PairwisePathSelector(
            model.n_digit,
            model.codebook_size,
            model.n_embd,
            rank=int(run_args.get('pair_rank', 32)),
            triple_rank=(
                int(run_args.get('triple_rank', 16)) if variant == 'triple' else 0
            ),
        ).to(device)
        selector.load_state_dict(checkpoint['selector'])

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    return model, selector, ar_model


@torch.no_grad()
def collect_model_outputs(
    model,
    selector,
    ar_model,
    loader,
    catalog,
    attributes,
    proposal_k,
    fusion_alpha,
    candidate_score_chunk_size,
    include_standalone_ar,
    standalone_top_k,
    save_representations,
    split,
):
    model.eval()
    if selector is not None:
        selector.eval()
    ar_model.eval()
    ar_model.config['current_split'] = split
    storage = {}
    candidate_storage = {}
    offset = 0
    started = time.perf_counter()

    for batch in tqdm(loader, desc='complementarity full-test forward'):
        labels = batch['labels'].to(catalog.device).long()
        batch_size = labels.shape[0]
        target_rows = code_rows(labels, catalog, model.codebook_size)
        expected_item_ids = torch.as_tensor(
            attributes['target_item_id'][offset:offset + batch_size],
            device=catalog.device,
        )
        if not torch.equal(target_rows + 1, expected_item_ids):
            raise ValueError('raw and tokenized target item order diverged')

        drafter_encoder_hidden = encode_history(model, batch)
        scores, logits, drafter_coordinate_hidden, unary, pairwise = one_pass_outputs(
            model, batch, catalog, selector,
            encoder_hidden=drafter_encoder_hidden,
        )
        keep = min(int(proposal_k), catalog.shape[0])
        proposal_scores, proposal_rows = torch.topk(scores, k=keep, dim=1)
        proposals = catalog[proposal_rows]
        drafter_rank = stable_target_ranks(scores, target_rows)
        unary_rank = stable_target_ranks(unary, target_rows)
        target_scores = scores.gather(1, target_rows[:, None]).squeeze(1)
        target_unary = unary.gather(1, target_rows[:, None]).squeeze(1)
        if pairwise is None:
            target_pairwise = torch.zeros_like(target_unary)
        else:
            target_pairwise = pairwise.gather(
                1, target_rows[:, None]
            ).squeeze(1)
        drafter_log_probs = F.log_softmax(logits.float(), dim=-1)
        drafter_token_logp = drafter_log_probs.gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)

        # Encode history once and score all proposed paths plus the target.
        history_sid = batch['history_sid'].to(catalog.device)
        ar_hidden = ar_model.forward(
            {'history_sid': history_sid}, return_loss=False
        ).hidden_states
        history_mask = history_sid.ne(-1).any(dim=-1)
        all_paths = torch.cat([proposals, labels[:, None, :]], dim=1)
        path_logits = ar_model.candidate_logits_from_encoded_history(
            ar_hidden,
            history_mask,
            all_paths,
            chunk_size=candidate_score_chunk_size,
        )
        path_log_probs = F.log_softmax(path_logits.float(), dim=-1)
        path_token_scores = path_log_probs.gather(
            -1, all_paths.unsqueeze(-1)
        ).squeeze(-1)
        ar_scores = path_token_scores[:, :keep].sum(dim=-1)
        target_ar_token_logp = path_token_scores[:, keep]
        target_ar_score = target_ar_token_logp.sum(dim=-1)

        ar_order = ar_scores.argsort(dim=1, descending=True)
        ar_rows = proposal_rows.gather(1, ar_order)
        ar_rank = rank_from_ordered_rows(ar_rows, target_rows)

        normalized_drafter = normalize_scores(proposal_scores)
        normalized_ar = normalize_scores(ar_scores)
        fused_scores = (
            (1.0 - fusion_alpha) * normalized_drafter
            + fusion_alpha * normalized_ar
        )
        fused_order = fused_scores.argsort(dim=1, descending=True)
        fused_rows = proposal_rows.gather(1, fused_order)
        fused_rank = rank_from_ordered_rows(fused_rows, target_rows)

        target_drafter_z = (
            target_scores - proposal_scores.mean(dim=1)
        ) / proposal_scores.std(dim=1, unbiased=False).clamp_min(1e-6)
        target_ar_z = (
            target_ar_score - ar_scores.mean(dim=1)
        ) / ar_scores.std(dim=1, unbiased=False).clamp_min(1e-6)
        target_fused_score = (
            (1.0 - fusion_alpha) * target_drafter_z
            + fusion_alpha * target_ar_z
        )
        ar_counterfactual_rank = target_counterfactual_rank(
            ar_scores, target_ar_score
        )
        fusion_counterfactual_rank = target_counterfactual_rank(
            fused_scores, target_fused_score
        )

        append_tensor(storage, 'target_codes', labels, np.int16)
        if save_representations:
            if drafter_encoder_hidden.shape[1] == 1:
                drafter_pooled = drafter_encoder_hidden[:, 0]
            else:
                last_index = history_mask.long().sum(dim=1).sub(1).clamp_min(0)
                drafter_pooled = drafter_encoder_hidden[
                    torch.arange(batch_size, device=catalog.device), last_index
                ]
            ar_last_index = history_mask.long().sum(dim=1).sub(1).clamp_min(0)
            ar_pooled = ar_hidden[
                torch.arange(batch_size, device=catalog.device), ar_last_index
            ]
            append_tensor(
                storage, 'drafter_history_state', drafter_pooled, np.float16
            )
            append_tensor(storage, 'ar_history_state', ar_pooled, np.float16)
            append_tensor(
                storage, 'drafter_coordinate_states',
                drafter_coordinate_hidden, np.float16,
            )
        append_tensor(storage, 'drafter_rank', drafter_rank, np.int32)
        append_tensor(storage, 'unary_rank', unary_rank, np.int32)
        append_tensor(storage, 'ar_rank', ar_rank, np.int16)
        append_tensor(storage, 'fusion_rank', fused_rank, np.int16)
        append_tensor(
            storage, 'ar_counterfactual_rank', ar_counterfactual_rank, np.int16
        )
        append_tensor(
            storage, 'fusion_counterfactual_rank',
            fusion_counterfactual_rank, np.int16,
        )
        append_tensor(storage, 'target_drafter_score', target_scores, np.float32)
        append_tensor(storage, 'target_unary_score', target_unary, np.float32)
        append_tensor(storage, 'target_pairwise_score', target_pairwise, np.float32)
        append_tensor(storage, 'target_ar_score', target_ar_score, np.float32)
        append_tensor(storage, 'target_drafter_z', target_drafter_z, np.float32)
        append_tensor(storage, 'target_ar_z', target_ar_z, np.float32)
        append_tensor(storage, 'target_fused_score', target_fused_score, np.float32)
        append_tensor(
            storage, 'target_drafter_token_logp', drafter_token_logp, np.float32
        )
        append_tensor(
            storage, 'drafter_token_correct',
            logits.argmax(dim=-1).eq(labels), np.bool_,
        )
        append_tensor(
            storage, 'target_ar_token_logp', target_ar_token_logp, np.float32
        )
        append_tensor(
            storage, 'ar_teacher_forced_token_correct',
            path_logits[:, keep].argmax(dim=-1).eq(labels), np.bool_,
        )
        append_tensor(
            storage,
            'drafter_target_margin',
            wrong_candidate_margin(
                proposal_scores, proposal_rows, target_rows, target_scores
            ),
            np.float32,
        )
        append_tensor(
            storage,
            'ar_target_margin',
            wrong_candidate_margin(
                ar_scores, proposal_rows, target_rows, target_ar_score
            ),
            np.float32,
        )
        append_tensor(
            storage,
            'fusion_target_margin',
            wrong_candidate_margin(
                fused_scores, proposal_rows, target_rows, target_fused_score
            ),
            np.float32,
        )
        append_tensor(
            storage,
            'drafter_ar_score_correlation',
            (normalized_drafter * normalized_ar).mean(dim=1),
            np.float32,
        )
        append_tensor(
            storage, 'drafter_candidate_entropy',
            normalized_entropy(proposal_scores), np.float32,
        )
        append_tensor(
            storage, 'ar_candidate_entropy',
            normalized_entropy(ar_scores), np.float32,
        )
        append_tensor(
            storage, 'drafter_ar_top10_overlap',
            topk_overlap(proposal_rows, ar_rows, 10), np.int8,
        )
        append_tensor(
            storage, 'drafter_fusion_top10_overlap',
            topk_overlap(proposal_rows, fused_rows, 10), np.int8,
        )
        append_tensor(
            storage, 'ar_fusion_top10_overlap',
            topk_overlap(ar_rows, fused_rows, 10), np.int8,
        )

        append_tensor(candidate_storage, 'proposal_item_ids', proposal_rows + 1, np.int32)
        append_tensor(candidate_storage, 'drafter_candidate_scores', proposal_scores, np.float32)
        append_tensor(
            candidate_storage, 'drafter_candidate_token_scores',
            drafter_log_probs[:, None].expand(
                -1, keep, -1, -1
            ).gather(-1, proposals.unsqueeze(-1)).squeeze(-1),
            np.float16,
        )
        append_tensor(
            candidate_storage, 'unary_candidate_scores',
            unary.gather(1, proposal_rows), np.float32,
        )
        if pairwise is not None:
            append_tensor(
                candidate_storage, 'pairwise_candidate_scores',
                pairwise.gather(1, proposal_rows), np.float32,
            )
        append_tensor(candidate_storage, 'ar_candidate_scores', ar_scores, np.float32)
        append_tensor(
            candidate_storage, 'ar_candidate_token_scores',
            path_token_scores[:, :keep], np.float16,
        )
        append_tensor(candidate_storage, 'fusion_candidate_scores', fused_scores, np.float32)

        if include_standalone_ar:
            generated = ar_model.generate(
                batch, n_return_sequences=standalone_top_k
            )
            generated_rows = code_rows(
                generated.reshape(-1, generated.shape[-1]),
                catalog,
                model.codebook_size,
            ).reshape(generated.shape[:2])
            standalone_rank = rank_from_ordered_rows(generated_rows, target_rows)
            append_tensor(storage, 'standalone_ar_rank', standalone_rank, np.int16)
            append_tensor(
                storage, 'standalone_drafter_top10_overlap',
                topk_overlap(generated_rows, proposal_rows, 10), np.int8,
            )
            append_tensor(
                storage, 'standalone_drafter_topk_overlap',
                set_overlap(generated_rows, proposal_rows),
                np.int8,
            )
            append_tensor(
                candidate_storage, 'standalone_ar_item_ids',
                generated_rows + 1, np.int32,
            )
        offset += batch_size

    elapsed = time.perf_counter() - started
    output = concatenate_storage(storage)
    candidates = concatenate_storage(candidate_storage)
    output['elapsed_seconds'] = np.asarray([elapsed], dtype=np.float64)
    output['milliseconds_per_example'] = np.asarray(
        [1000.0 * elapsed / max(offset, 1)], dtype=np.float64
    )
    return output, candidates


def interval_labels(values, boundaries, names):
    values = np.asarray(values)
    if len(names) != len(boundaries) + 1:
        raise ValueError('interval names must have one more entry than boundaries')
    indices = np.digitize(values, boundaries, right=True)
    return np.asarray(names, dtype=object)[indices]


def quantile_labels(values, prefix, n_bins=4):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.full(values.shape, 'missing', dtype=object)
    if not finite.any():
        return output, []
    boundaries = np.unique(
        np.quantile(values[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
    )
    indices = np.digitize(values[finite], boundaries, right=True)
    names = [f'{prefix}_q{i + 1}' for i in range(len(boundaries) + 1)]
    output[finite] = np.asarray(names, dtype=object)[indices]
    return output, boundaries.tolist()


def summarize_group(labels, arrays):
    rows = []
    drafter_hit = (arrays['drafter_rank'] > 0) & (arrays['drafter_rank'] <= 10)
    ar_hit = (arrays['ar_rank'] > 0) & (arrays['ar_rank'] <= 10)
    fusion_hit = (arrays['fusion_rank'] > 0) & (arrays['fusion_rank'] <= 10)
    candidate_hit = arrays['drafter_rank'] <= int(arrays['proposal_k'][0])
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        count = int(mask.sum())
        if not count:
            continue
        row = {
            'group': str(label),
            'count': count,
            'fraction': float(mask.mean()),
            'candidate_recall': float(candidate_hit[mask].mean()),
            'drafter_recall@10': float(drafter_hit[mask].mean()),
            'ar_recall@10': float(ar_hit[mask].mean()),
            'fusion_recall@10': float(fusion_hit[mask].mean()),
            'fusion_minus_drafter': float(
                fusion_hit[mask].mean() - drafter_hit[mask].mean()
            ),
            'fusion_minus_ar': float(
                fusion_hit[mask].mean() - ar_hit[mask].mean()
            ),
            'both_rate': float((drafter_hit[mask] & ar_hit[mask]).mean()),
            'drafter_only_rate': float((drafter_hit[mask] & ~ar_hit[mask]).mean()),
            'ar_only_rate': float((~drafter_hit[mask] & ar_hit[mask]).mean()),
            'neither_rate': float((~drafter_hit[mask] & ~ar_hit[mask]).mean()),
            'fusion_rescue_both_miss_rate': float(
                (fusion_hit[mask] & ~drafter_hit[mask] & ~ar_hit[mask]).mean()
            ),
            'fusion_harm_union_rate': float(
                (~fusion_hit[mask] & (drafter_hit[mask] | ar_hit[mask])).mean()
            ),
        }
        rows.append(row)
    return rows


def build_summary(arrays, expected, fusion_alpha, proposal_k, protocol):
    arrays['proposal_k'] = np.full(
        arrays['drafter_rank'].shape, proposal_k, dtype=np.int16
    )
    metrics = {
        'drafter': rank_metrics(arrays['drafter_rank']),
        'ar_verified': rank_metrics(arrays['ar_rank']),
        'fusion': rank_metrics(arrays['fusion_rank']),
    }
    if 'unary_rank' in arrays:
        metrics['unary_only'] = rank_metrics(arrays['unary_rank'])
    if 'standalone_ar_rank' in arrays:
        metrics['standalone_ar'] = rank_metrics(arrays['standalone_ar_rank'])

    d = (arrays['drafter_rank'] > 0) & (arrays['drafter_rank'] <= 10)
    a = (arrays['ar_rank'] > 0) & (arrays['ar_rank'] <= 10)
    f = (arrays['fusion_rank'] > 0) & (arrays['fusion_rank'] <= 10)
    quadrant = {
        'both': int((d & a).sum()),
        'drafter_only': int((d & ~a).sum()),
        'ar_only': int((~d & a).sum()),
        'neither': int((~d & ~a).sum()),
        'oracle_union_recall@10': float((d | a).mean()),
        'fusion_rescued_both_miss': int((f & ~d & ~a).sum()),
        'fusion_harmed_union': int((~f & (d | a)).sum()),
        'fusion_hit@10': int(f.sum()),
    }

    rank_bucket = interval_labels(
        arrays['drafter_rank'], [10, 32, proposal_k],
        ['rank_1_10', 'rank_11_32', f'rank_33_{proposal_k}', f'rank_gt_{proposal_k}'],
    )
    popularity = interval_labels(
        arrays['target_train_count'], [0, 1, 3, 10, 50],
        ['pop_0', 'pop_1', 'pop_2_3', 'pop_4_10', 'pop_11_50', 'pop_51_plus'],
    )
    transition = interval_labels(
        arrays['last_transition_count'], [0, 1, 3, 10],
        ['transition_0', 'transition_1', 'transition_2_3', 'transition_4_10', 'transition_11_plus'],
    )
    history = interval_labels(
        arrays['history_length'], [5, 10, 20],
        ['history_1_5', 'history_6_10', 'history_11_20', 'history_21_plus'],
    )
    repeat = np.where(arrays['repeat_target'], 'repeat', 'novel')
    sid_hamming = np.asarray(
        [f'sid_min_hamming_{int(value)}' for value in arrays['sid_min_hamming']],
        dtype=object,
    )
    text_quantile, text_boundaries = quantile_labels(
        arrays['text_max_cosine'], 'text_max_cosine'
    )
    correlation_quantile, correlation_boundaries = quantile_labels(
        arrays['drafter_ar_score_correlation'], 'score_corr'
    )
    groups = {
        'drafter_rank_bucket': summarize_group(rank_bucket, arrays),
        'target_popularity': summarize_group(popularity, arrays),
        'last_transition_frequency': summarize_group(transition, arrays),
        'history_length': summarize_group(history, arrays),
        'repeat_novel': summarize_group(repeat, arrays),
        'sid_min_hamming': summarize_group(sid_hamming, arrays),
        'text_max_cosine_quantile': summarize_group(text_quantile, arrays),
        'score_correlation_quantile': summarize_group(
            correlation_quantile, arrays
        ),
    }

    parity = {}
    if expected is not None:
        tag = alpha_tag(fusion_alpha)
        expected_test = expected['test']
        mapping = {
            'drafter_ndcg@5': metrics['drafter']['ndcg@5'],
            'drafter_recall@5': metrics['drafter']['recall@5'],
            'drafter_ndcg@10': metrics['drafter']['ndcg@10'],
            'drafter_recall@10': metrics['drafter']['recall@10'],
            'ar_verified_ndcg@5': metrics['ar_verified']['ndcg@5'],
            'ar_verified_recall@5': metrics['ar_verified']['recall@5'],
            'ar_verified_ndcg@10': metrics['ar_verified']['ndcg@10'],
            'ar_verified_recall@10': metrics['ar_verified']['recall@10'],
            f'fused_a{tag}_ndcg@5': metrics['fusion']['ndcg@5'],
            f'fused_a{tag}_recall@5': metrics['fusion']['recall@5'],
            f'fused_a{tag}_ndcg@10': metrics['fusion']['ndcg@10'],
            f'fused_a{tag}_recall@10': metrics['fusion']['recall@10'],
        }
        for name, observed in mapping.items():
            if name in expected_test:
                reference = float(expected_test[name])
                parity[name] = {
                    'expected': reference,
                    'observed': float(observed),
                    'absolute_error': abs(float(observed) - reference),
                }

    return {
        'protocol': protocol,
        'metrics': metrics,
        'quadrants': quadrant,
        'descriptives': {
            'mean_score_correlation': float(
                np.mean(arrays['drafter_ar_score_correlation'])
            ),
            'mean_drafter_ar_top10_overlap': float(
                np.mean(arrays['drafter_ar_top10_overlap'])
            ),
            'mean_drafter_fusion_top10_overlap': float(
                np.mean(arrays['drafter_fusion_top10_overlap'])
            ),
            'mean_ar_fusion_top10_overlap': float(
                np.mean(arrays['ar_fusion_top10_overlap'])
            ),
            'mean_drafter_candidate_entropy': float(
                np.mean(arrays['drafter_candidate_entropy'])
            ),
            'mean_ar_candidate_entropy': float(
                np.mean(arrays['ar_candidate_entropy'])
            ),
            'text_max_cosine_quantile_boundaries': text_boundaries,
            'score_correlation_quantile_boundaries': correlation_boundaries,
        },
        'groups': groups,
        'metric_parity': parity,
    }


def markdown_table(rows, columns):
    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join(['---'] + ['---:' for _ in columns[1:]]) + ' |'
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f'{value:.6f}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def render_markdown(summary):
    protocol = summary['protocol']
    metrics = summary['metrics']
    quadrants = summary['quadrants']
    lines = [
        '# Drafter / AR complementarity analysis',
        '',
        f"Dataset: `{protocol['dataset']}`; split: `{protocol['split']}`; "
        f"examples: {protocol['n_examples']}; K={protocol['proposal_k']}; "
        f"fusion alpha={protocol['fusion_alpha']}.",
        '',
        '## Aggregate parity metrics',
        '',
    ]
    metric_rows = []
    for method, values in metrics.items():
        metric_rows.append({'method': method, **values})
    lines.append(markdown_table(
        metric_rows,
        ['method', 'ndcg@5', 'recall@5', 'ndcg@10', 'recall@10'],
    ))
    lines.extend([
        '',
        '## Same-candidate Top-10 quadrants',
        '',
        f"- both correct: {quadrants['both']}",
        f"- drafter only: {quadrants['drafter_only']}",
        f"- AR only: {quadrants['ar_only']}",
        f"- neither: {quadrants['neither']}",
        f"- per-example oracle union Recall@10: "
        f"{quadrants['oracle_union_recall@10']:.6f}",
        f"- fusion rescues when both individual ranks miss: "
        f"{quadrants['fusion_rescued_both_miss']}",
        f"- fusion harms an item found by either component: "
        f"{quadrants['fusion_harmed_union']}",
        '',
        '## Diagnostic groups',
        '',
    ])
    columns = [
        'group', 'count', 'candidate_recall', 'drafter_recall@10',
        'ar_recall@10', 'fusion_recall@10', 'fusion_minus_drafter',
        'fusion_minus_ar',
    ]
    for name, rows in summary['groups'].items():
        lines.extend([f'### {name}', '', markdown_table(rows, columns), ''])
    lines.extend([
        '## Reproducibility',
        '',
        f"- drafter checkpoint: `{protocol['drafter_checkpoint']}`",
        f"- AR checkpoint: `{protocol['ar_checkpoint']}`",
        f"- expected result: `{protocol.get('expected_result')}`",
        '- `examples.npz` is the reusable per-example source artifact.',
    ])
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    drafter_checkpoint_path = resolve_path(args.drafter_checkpoint)
    checkpoint = torch.load(
        drafter_checkpoint_path, map_location='cpu', weights_only=False
    )
    run_args = dict(checkpoint['args'])
    expected_path = resolve_path(args.expected_result)
    if expected_path is None:
        candidate = drafter_checkpoint_path.parent / 'result.json'
        expected_path = candidate if candidate.is_file() else None
    expected = load_json(expected_path) if expected_path is not None else None
    proposal_k = int(args.proposal_k or run_args.get('proposal_k', 72))
    fusion_alpha = args.fusion_alpha
    if fusion_alpha is None and expected is not None:
        fusion_alpha = expected.get('selected_fusion_alpha')
    if fusion_alpha is None:
        fusion_alpha = 0.75
    fusion_alpha = float(fusion_alpha)
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError('fusion alpha must lie in [0,1]')

    ar_checkpoint_path = resolve_path(
        args.ar_checkpoint or run_args['ar_checkpoint']
    )
    common_files = [str(resolve_path(run_args['common_config']))]
    if run_args.get('sid_config'):
        common_files.append(str(resolve_path(run_args['sid_config'])))
    accelerator = Accelerator()
    diffusion_config = make_config(
        'DIFF_GRM', run_args['dataset'],
        common_files + [str(resolve_path(run_args['diffusion_config']))],
        accelerator,
        {'eval_batch_size': args.batch_size},
    )
    ar_config = make_config(
        'AR_GRM', run_args['dataset'],
        common_files + [str(resolve_path(run_args['ar_config']))],
        accelerator,
        {
            'eval_batch_size': args.batch_size,
            'candidate_score_chunk_size': args.candidate_score_chunk_size,
        },
    )
    ar_config['current_split'] = args.split
    device = torch.device(diffusion_config['device'])
    dataset = get_dataset(run_args['dataset'])(diffusion_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    raw_splits = dataset.split()
    raw_split = raw_splits[args.split]
    if args.max_examples is not None:
        maximum = min(int(args.max_examples), len(raw_split))
        raw_split = raw_split.select(range(maximum))
    tokenized = tokenizer.tokenize({args.split: raw_split})[args.split]
    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn[args.split],
    )
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('complementarity analysis requires collision-free SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    model, selector, ar_model = instantiate_models(
        checkpoint,
        run_args,
        diffusion_config,
        ar_config,
        dataset,
        tokenizer,
        device,
    )
    ar_model.load_state_dict(
        torch.load(ar_checkpoint_path, map_location=device, weights_only=False)
    )

    embeddings, embedding_path = load_text_embeddings(
        dataset, tokenizer, diffusion_config
    )
    attributes, history_ids = build_example_attributes(
        raw_splits['train'],
        raw_split,
        dataset,
        raw_catalog,
        embeddings,
        int(diffusion_config['max_history_len']),
    )
    n_examples = len(raw_split)

    outputs, candidate_arrays = collect_model_outputs(
        model,
        selector,
        ar_model,
        loader,
        catalog,
        attributes,
        proposal_k,
        fusion_alpha,
        args.candidate_score_chunk_size,
        args.include_standalone_ar,
        args.standalone_top_k,
        args.save_representations,
        args.split,
    )
    arrays = {**attributes, **outputs}
    arrays['history_item_ids'] = history_ids

    protocol = {
        'dataset': run_args['dataset'],
        'category': diffusion_config.get('category'),
        'split': args.split,
        'n_examples': n_examples,
        'catalog_items': int(raw_catalog.shape[0]),
        'n_digit': int(raw_catalog.shape[1]),
        'codebook_size': int(ar_config['codebook_size']),
        'max_history_len': int(diffusion_config['max_history_len']),
        'proposal_k': proposal_k,
        'fusion_alpha': fusion_alpha,
        'include_standalone_ar': bool(args.include_standalone_ar),
        'save_representations': bool(args.save_representations),
        'drafter_checkpoint': str(drafter_checkpoint_path),
        'ar_checkpoint': str(ar_checkpoint_path),
        'expected_result': None if expected_path is None else str(expected_path),
        'text_embedding_cache': str(embedding_path),
        'elapsed_seconds': float(outputs['elapsed_seconds'][0]),
        'milliseconds_per_example': float(outputs['milliseconds_per_example'][0]),
    }
    # Aggregate parity is meaningful only for the complete canonical split.
    # Keeping the reference path in ``protocol`` is still useful for provenance,
    # but comparing a diagnostic subset against full-test metrics is misleading.
    parity_reference = (
        expected
        if args.max_examples is None and args.split == 'test'
        else None
    )
    summary = build_summary(
        arrays, parity_reference, fusion_alpha, proposal_k, protocol
    )
    parity_errors = [
        row['absolute_error'] for row in summary['metric_parity'].values()
    ]
    if (
        args.max_examples is None
        and parity_errors
        and max(parity_errors) > args.parity_atol
    ):
        raise RuntimeError(
            f'aggregate parity failed: max error={max(parity_errors):.3g} '
            f'> tolerance={args.parity_atol:.3g}'
        )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_to_save = dict(arrays)
    arrays_to_save['proposal_k'] = np.asarray([proposal_k], dtype=np.int16)
    arrays_to_save['fusion_alpha'] = np.asarray([fusion_alpha], dtype=np.float32)
    if not args.no_candidate_matrix:
        arrays_to_save.update(candidate_arrays)
    np.savez_compressed(output_dir / 'examples.npz', **arrays_to_save)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    (output_dir / 'REPORT.md').write_text(
        render_markdown(summary), encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'metrics': summary['metrics'],
        'quadrants': summary['quadrants'],
        'max_metric_parity_error': max(parity_errors, default=None),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
