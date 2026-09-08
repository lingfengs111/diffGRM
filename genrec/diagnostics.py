"""Shared diagnostics for semantic-ID representations and decoders.

The training metrics answer whether an item was retrieved.  These diagnostics
answer *why* it was or was not retrieved: where a decoded path first diverged,
whether a decoder can predict each code under controlled conditioning, and how
information is distributed across the catalog code dimensions.
"""

from collections import OrderedDict
from itertools import combinations
import math

import numpy as np
import torch
import torch.nn.functional as F


def coordinate_subset_masks(n_digit, include_empty=True, include_full=True):
    """Enumerate coordinate subsets as integer masks in cardinality order.

    Cardinality ordering makes diagnostic tables easier to read while the
    integer mask remains suitable as a stable result key.
    """
    if int(n_digit) <= 0:
        raise ValueError(f"n_digit must be positive, got {n_digit}")
    masks = []
    for size in range(int(n_digit) + 1):
        for digits in combinations(range(int(n_digit)), size):
            mask = sum(1 << digit for digit in digits)
            if mask == 0 and not include_empty:
                continue
            if mask == (1 << int(n_digit)) - 1 and not include_full:
                continue
            masks.append(mask)
    return masks


def subset_digits(mask, n_digit):
    """Return the sorted coordinates present in an integer subset mask."""
    return [digit for digit in range(int(n_digit)) if int(mask) & (1 << digit)]


def subset_key(mask, n_digit):
    digits = subset_digits(mask, n_digit)
    return "none" if not digits else "-".join(str(digit) for digit in digits)


def stable_target_ranks(scores, target_rows):
    """Return one-based ranks with catalog-row order as a deterministic tie break.

    Args:
        scores: ``[batch, catalog]`` candidate scores; illegal rows may be -inf.
        target_rows: ``[batch]`` catalog row of each ground-truth item.
    """
    if scores.ndim != 2 or target_rows.ndim != 1:
        raise ValueError("expected scores [B,N] and target_rows [B]")
    if scores.shape[0] != target_rows.shape[0]:
        raise ValueError("score and target batch sizes do not match")
    target_rows = target_rows.to(scores.device).long()
    target_scores = scores.gather(1, target_rows[:, None])
    catalog_rows = torch.arange(scores.shape[1], device=scores.device)[None, :]
    strictly_better = scores.gt(target_scores)
    tied_before = scores.eq(target_scores) & catalog_rows.lt(target_rows[:, None])
    return 1 + (strictly_better | tied_before).sum(dim=1)


def catalog_subset_diagnostics(codes):
    """Measure information allocation and candidate buckets for every subset.

    Unlike prefix-only statistics, this treats product-code coordinates as an
    unordered set and therefore supports OPQ/PQ diagnostics directly.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"codes must be rank 2, got {codes.shape}")
    n_items, n_digit = codes.shape
    if n_items == 0:
        raise ValueError("catalog is empty")

    coordinate_entropy = [_discrete_entropy(codes[:, digit]) for digit in range(n_digit)]
    pairwise_nmi = {}
    for left in range(n_digit):
        for right in range(left + 1, n_digit):
            joint = _discrete_entropy(codes[:, [left, right]])
            mutual_information = coordinate_entropy[left] + coordinate_entropy[right] - joint
            denom = math.sqrt(max(coordinate_entropy[left] * coordinate_entropy[right], 1e-12))
            pairwise_nmi[f"{left}-{right}"] = float(mutual_information / denom)

    subsets = OrderedDict()
    for mask in coordinate_subset_masks(n_digit, include_empty=False, include_full=True):
        digits = subset_digits(mask, n_digit)
        _, inverse, counts = np.unique(
            codes[:, digits], axis=0, return_inverse=True, return_counts=True
        )
        item_bucket_sizes = counts[inverse]
        subsets[subset_key(mask, n_digit)] = {
            "mask": int(mask),
            "digits": digits,
            "num_groups": int(len(counts)),
            "mean_group_size_over_groups": float(counts.mean()),
            "mean_candidate_count_per_item": float(item_bucket_sizes.mean()),
            "median_candidate_count_per_item": float(np.median(item_bucket_sizes)),
            "p90_candidate_count_per_item": float(np.percentile(item_bucket_sizes, 90)),
            "p99_candidate_count_per_item": float(np.percentile(item_bucket_sizes, 99)),
            "singleton_item_rate": float(np.mean(item_bucket_sizes == 1)),
            "joint_entropy_nats": _discrete_entropy(codes[:, digits]),
        }

    return {
        "n_items": int(n_items),
        "n_digit": int(n_digit),
        "full_unique_ratio": float(len(np.unique(codes, axis=0)) / n_items),
        "coordinate_entropy_nats": coordinate_entropy,
        "pairwise_normalized_mutual_information": pairwise_nmi,
        "subsets": subsets,
    }


def generation_diagnostics(preds, labels, prefix="diag/free"):
    """Return per-example top-1 path diagnostics.

    Args:
        preds: ``[batch, beam, digit]`` raw codebook IDs.
        labels: ``[batch, digit]`` raw codebook IDs.
    """
    if preds.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"expected preds [B,K,L], labels [B,L], got {preds.shape}, {labels.shape}")
    if preds.shape[0] != labels.shape[0] or preds.shape[2] != labels.shape[1]:
        raise ValueError("prediction and label dimensions do not match")

    top1 = preds[:, 0].to(labels.device)
    correct = top1.eq(labels)
    results = OrderedDict()
    prefix_correct = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
    for digit in range(labels.shape[1]):
        prefix_correct = prefix_correct & correct[:, digit]
        results[f"{prefix}/digit{digit}_acc"] = correct[:, digit].float()
        results[f"{prefix}/prefix{digit + 1}_acc"] = prefix_correct.float()
        first_error = (~correct[:, digit]) & (
            correct[:, :digit].all(dim=1) if digit else torch.ones_like(correct[:, digit])
        )
        results[f"{prefix}/first_error_digit{digit}"] = first_error.float()
    results[f"{prefix}/full_sid_acc"] = correct.all(dim=1).float()
    results[f"{prefix}/no_error"] = correct.all(dim=1).float()
    return results


def conditional_diagnostics(logits, labels, prefix="diag/conditional"):
    """Return per-example, per-digit diagnostics for controlled logits.

    ``logits`` must have shape ``[batch, digit, codebook]``.  Unlike generated
    paths, these scores expose local prediction quality without compounding an
    earlier decoding error.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"expected logits [B,L,K], labels [B,L], got {logits.shape}, {labels.shape}")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit and label dimensions do not match")

    labels = labels.to(logits.device).long()
    if bool(((labels < 0) | (labels >= logits.shape[-1])).any()):
        raise ValueError("labels contain IDs outside the decoder codebook")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    target_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    target_logits = logits.gather(-1, labels.unsqueeze(-1))
    ranks = logits.gt(target_logits).sum(dim=-1) + 1
    entropy = -(probs * log_probs).sum(dim=-1) / math.log(logits.shape[-1])
    predictions = logits.argmax(dim=-1)

    results = OrderedDict()
    for digit in range(labels.shape[1]):
        results[f"{prefix}/digit{digit}_acc"] = predictions[:, digit].eq(labels[:, digit]).float()
        results[f"{prefix}/digit{digit}_nll"] = -target_logp[:, digit]
        results[f"{prefix}/digit{digit}_mrr"] = ranks[:, digit].float().reciprocal()
        results[f"{prefix}/digit{digit}_norm_entropy"] = entropy[:, digit]
    return results


def catalog_codes(tokenizer, codebook_size):
    """Recover raw catalog codes in item-id order from a model tokenizer."""
    rows = []
    for item, item_id in sorted(tokenizer.dataset.item2id.items(), key=lambda pair: pair[1]):
        if int(item_id) == 0:
            continue
        tokens = tokenizer.item2tokens[item]
        rows.append([
            int(token) - (int(tokenizer.sid_offset) + digit * int(codebook_size))
            for digit, token in enumerate(tokens)
        ])
    codes = np.asarray(rows, dtype=np.int64)
    if codes.ndim != 2 or codes.shape[1] != int(tokenizer.n_digit):
        raise ValueError(f"invalid catalog code shape: {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= int(codebook_size)):
        raise ValueError("catalog contains raw IDs outside the codebook")
    return codes


def _discrete_entropy(rows):
    """Empirical entropy in nats for a one- or multi-column discrete array."""
    rows = np.asarray(rows)
    if rows.ndim == 1:
        _, counts = np.unique(rows, return_counts=True)
    else:
        _, counts = np.unique(rows, axis=0, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def catalog_diagnostics(tokenizer, codebook_size, prefix="diag/catalog"):
    """Measure collisions and information allocation of catalog SIDs."""
    codes = catalog_codes(tokenizer, codebook_size)
    n_items, n_digit = codes.shape
    full_unique, full_counts = np.unique(codes, axis=0, return_counts=True)
    colliding_items = int(full_counts[full_counts > 1].sum())
    log_k = math.log(int(codebook_size))

    results = OrderedDict()
    results[f"{prefix}/n_items"] = float(n_items)
    results[f"{prefix}/unique_sid_ratio"] = float(len(full_unique) / max(n_items, 1))
    results[f"{prefix}/collision_excess_ratio"] = float((n_items - len(full_unique)) / max(n_items, 1))
    results[f"{prefix}/colliding_item_ratio"] = float(colliding_items / max(n_items, 1))
    results[f"{prefix}/max_collision_multiplicity"] = float(full_counts.max(initial=0))

    previous_joint_entropy = 0.0
    for digit in range(n_digit):
        values, counts = np.unique(codes[:, digit], return_counts=True)
        digit_entropy = _discrete_entropy(codes[:, digit])
        joint_entropy = _discrete_entropy(codes[:, :digit + 1])
        conditional_entropy = joint_entropy - previous_joint_entropy
        previous_joint_entropy = joint_entropy
        prefix_unique = len(np.unique(codes[:, :digit + 1], axis=0))
        results[f"{prefix}/digit{digit}_utilization"] = float(len(values) / int(codebook_size))
        results[f"{prefix}/digit{digit}_norm_entropy"] = float(digit_entropy / log_k)
        results[f"{prefix}/digit{digit}_conditional_norm_entropy"] = float(conditional_entropy / log_k)
        results[f"{prefix}/prefix{digit + 1}_unique_ratio"] = float(prefix_unique / max(n_items, 1))
        results[f"{prefix}/digit{digit}_max_bucket_ratio"] = float(counts.max() / max(n_items, 1))
    return results
