"""Order-marginalized complete-SID training utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import torch


def build_hard_negative_table(
    catalog_codes: np.ndarray,
    num_negatives: int,
    min_shared_digits: int = 2,
    seed: int = 2026,
) -> np.ndarray:
    """Precompute catalog negatives, prioritizing maximum code overlap."""
    codes = np.asarray(catalog_codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"catalog_codes must be rank-2, got {codes.shape}")
    if len(set(map(tuple, codes.tolist()))) != codes.shape[0]:
        raise ValueError("path-level item loss requires injective catalog SIDs")
    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")

    n_items, n_digit = codes.shape
    inverted: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for item_idx, row in enumerate(codes):
        for digit, code in enumerate(row):
            inverted[(digit, int(code))].append(item_idx)

    rng = np.random.default_rng(seed)
    result = np.empty((n_items, num_negatives), dtype=np.int64)
    all_items = np.arange(n_items, dtype=np.int64)
    for item_idx, row in enumerate(codes):
        overlap: Counter[int] = Counter()
        for digit, code in enumerate(row):
            overlap.update(inverted[(digit, int(code))])
        overlap.pop(item_idx, None)

        selected: list[int] = []
        for shared in range(n_digit - 1, min_shared_digits - 1, -1):
            same_level = [idx for idx, count in overlap.items() if count == shared]
            if same_level:
                rng.shuffle(same_level)
                selected.extend(same_level[: num_negatives - len(selected)])
            if len(selected) >= num_negatives:
                break

        if len(selected) < num_negatives:
            used = set(selected)
            used.add(item_idx)
            fallback = all_items[[idx not in used for idx in all_items]]
            need = num_negatives - len(selected)
            chosen = rng.choice(fallback, size=need, replace=len(fallback) < need)
            selected.extend(int(idx) for idx in np.atleast_1d(chosen))

        result[item_idx] = np.asarray(selected[:num_negatives], dtype=np.int64)
    return result


def order_marginal_dp(transition_log_probs: torch.Tensor) -> torch.Tensor:
    """Sum all reveal-order path probabilities with subset dynamic programming.

    Args:
        transition_log_probs: ``[..., 2**L - 1, L]``. Entry ``[..., S, d]``
            is log p(y_d | visible subset S, history), used only when ``d`` is
            not yet present in subset ``S``.

    Returns:
        ``[...]`` log-mean probability over all ``L!`` reveal orders.
    """
    if transition_log_probs.ndim < 2:
        raise ValueError("transition_log_probs must include state and digit axes")
    n_states_minus_full = transition_log_probs.shape[-2]
    n_digit = transition_log_probs.shape[-1]
    expected = (1 << n_digit) - 1
    if n_states_minus_full != expected:
        raise ValueError(
            f"expected {expected} predecessor states for L={n_digit}, "
            f"got {n_states_minus_full}"
        )

    full_state = (1 << n_digit) - 1
    dp: list[torch.Tensor | None] = [None] * (full_state + 1)
    dp[0] = torch.zeros_like(transition_log_probs[..., 0, 0])
    for state in range(full_state):
        if dp[state] is None:
            continue
        for digit in range(n_digit):
            if state & (1 << digit):
                continue
            next_state = state | (1 << digit)
            candidate = dp[state] + transition_log_probs[..., state, digit]
            if dp[next_state] is None:
                dp[next_state] = candidate
            else:
                dp[next_state] = torch.logaddexp(dp[next_state], candidate)

    return dp[full_state] - torch.lgamma(
        torch.tensor(float(n_digit + 1), device=transition_log_probs.device)
    )


def subset_masks(n_digit: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return visible and masked flags for every non-full subset state."""
    states = torch.arange((1 << n_digit) - 1, device=device, dtype=torch.long)
    digits = torch.arange(n_digit, device=device, dtype=torch.long)
    visible = (states[:, None] & (1 << digits[None, :])) != 0
    return visible, ~visible
