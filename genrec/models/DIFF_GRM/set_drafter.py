"""Typed-set masking utilities for semantic-ID proposal models.

The coordinates of an OPQ semantic ID have distinct codebooks, so they are
typed rather than fully exchangeable.  Their *reveal order*, however, need not
be fixed.  The helpers in this module train and score arbitrary observed/masked
coordinate subsets without introducing a left-to-right order.
"""

from itertools import combinations

import torch
import torch.nn.functional as F


def typed_subset_patterns(
    n_digit: int,
    min_masked: int = 1,
    max_masked: int | None = None,
    device=None,
) -> torch.Tensor:
    """Return every allowed non-empty typed-coordinate mask.

    ``True`` means that the coordinate is hidden and must be denoised.  Each
    coordinate keeps its identity; only the subset/reveal order is flexible.
    """
    n_digit = int(n_digit)
    min_masked = int(min_masked)
    max_masked = n_digit if max_masked is None else int(max_masked)
    if n_digit <= 0 or not 1 <= min_masked <= max_masked <= n_digit:
        raise ValueError(
            'expected 1 <= min_masked <= max_masked <= n_digit'
        )
    patterns = []
    for width in range(min_masked, max_masked + 1):
        for digits in combinations(range(n_digit), width):
            mask = [False] * n_digit
            for digit in digits:
                mask[digit] = True
            patterns.append(mask)
    return torch.tensor(patterns, dtype=torch.bool, device=device)


def sample_typed_subset_masks(
    batch_size: int,
    n_digit: int,
    n_views: int = 1,
    min_masked: int = 1,
    max_masked: int | None = None,
    device=None,
) -> torch.Tensor:
    """Uniformly sample typed mask patterns with shape ``[B, V, D]``."""
    patterns = typed_subset_patterns(
        n_digit,
        min_masked=min_masked,
        max_masked=max_masked,
        device=device,
    )
    rows = torch.randint(
        patterns.shape[0],
        (int(batch_size), int(n_views)),
        device=patterns.device,
    )
    return patterns[rows]


def conditional_catalog_scores(
    logits: torch.Tensor,
    catalog_codes: torch.Tensor,
    observed_codes: torch.Tensor,
    mask_positions: torch.Tensor,
    structural_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score legal items conditioned on an arbitrary observed coordinate set.

    Only hidden coordinates contribute token log-probabilities.  Catalog rows
    inconsistent with an observed coordinate receive ``-inf``.  Optional
    structural scores (for example pair/triple factors) are added before the
    legality mask is applied.
    """
    if logits.ndim != 3 or catalog_codes.ndim != 2:
        raise ValueError('expected logits [B,D,K] and catalog [N,D]')
    if observed_codes.shape != mask_positions.shape:
        raise ValueError('observed codes and mask positions must match')
    if logits.shape[:2] != observed_codes.shape:
        raise ValueError('batch/digit dimensions do not match')
    if catalog_codes.shape[1] != logits.shape[1]:
        raise ValueError('catalog and logits use different digit counts')

    hidden = mask_positions.bool()
    log_probs = F.log_softmax(logits, dim=-1)
    scores = logits.new_zeros(logits.shape[0], catalog_codes.shape[0])
    valid = torch.ones(
        logits.shape[0],
        catalog_codes.shape[0],
        dtype=torch.bool,
        device=logits.device,
    )
    for digit in range(logits.shape[1]):
        digit_scores = log_probs[:, digit, catalog_codes[:, digit]]
        scores = scores + digit_scores * hidden[:, digit, None]
        matches = catalog_codes[None, :, digit].eq(
            observed_codes[:, None, digit]
        )
        valid = valid & (hidden[:, digit, None] | matches)
    if structural_scores is not None:
        if structural_scores.shape != scores.shape:
            raise ValueError('structural scores must have shape [B,N]')
        scores = scores + structural_scores
    return scores.masked_fill(~valid, float('-inf')), valid


def masked_standardize(
    scores: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Row-wise standardization that ignores invalid candidate entries."""
    if scores.shape != valid.shape:
        raise ValueError('scores and valid mask must have identical shapes')
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    safe = scores.masked_fill(~valid, 0.0)
    mean = safe.sum(dim=1, keepdim=True) / count
    centered = torch.where(valid, scores - mean, torch.zeros_like(scores))
    variance = centered.square().sum(dim=1, keepdim=True) / count
    normalized = (scores - mean) / variance.sqrt().clamp_min(eps)
    return normalized.masked_fill(~valid, float('-inf'))
