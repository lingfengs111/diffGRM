"""Collision handling utilities for parallel/product semantic IDs.

The native OPQ/PQ objective minimizes quantization distortion; it does not
require the product code assigned to each catalog item to be injective.  The
helpers here support two controlled alternatives:

* append a TIGER-style collision resolver digit; and
* keep the original number of PQ digits while solving a minimum-distortion
  one-coordinate assignment problem that makes every full code unique.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def collision_stats(codes: np.ndarray) -> dict[str, Any]:
    """Return catalog collision statistics for an ``[N, L]`` code array."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"codes must be rank-2, got shape={codes.shape}")

    counts = Counter(map(tuple, codes.tolist()))
    colliding_sizes = [size for size in counts.values() if size > 1]
    involved = sum(colliding_sizes)
    duplicate_excess = int(codes.shape[0] - len(counts))
    return {
        "num_items": int(codes.shape[0]),
        "num_unique_codes": int(len(counts)),
        "num_collision_groups": int(len(colliding_sizes)),
        "num_items_in_collision_groups": int(involved),
        "collision_item_rate": float(involved / max(1, codes.shape[0])),
        "duplicate_excess": duplicate_excess,
        "duplicate_excess_rate": float(duplicate_excess / max(1, codes.shape[0])),
        "max_collision_group_size": int(max(colliding_sizes, default=1)),
    }


def append_dedup_digit(codes: np.ndarray, codebook_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Append a deterministic within-collision-group resolver digit."""
    codes = np.asarray(codes, dtype=np.int64)
    seen: defaultdict[tuple[int, ...], int] = defaultdict(int)
    resolver = np.zeros(codes.shape[0], dtype=np.int64)
    for row_idx, row in enumerate(codes):
        key = tuple(int(value) for value in row)
        resolver[row_idx] = seen[key]
        seen[key] += 1

    required = int(resolver.max(initial=0) + 1)
    if required > int(codebook_size):
        raise ValueError(
            f"dedup digit needs {required} values, but codebook_size={codebook_size}"
        )

    result = np.concatenate([codes, resolver[:, None]], axis=1)
    report = {
        "strategy": "append_dedup",
        "resolver_values_used": required,
        "before": collision_stats(codes),
        "after": collision_stats(result),
    }
    if report["after"]["duplicate_excess"] != 0:
        raise RuntimeError("append_dedup_digit failed to produce injective codes")
    return result, report


def _repair_one_digit(
    codes: np.ndarray,
    pq_subvectors: np.ndarray,
    centroids: np.ndarray,
    repair_digit: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair collisions by assigning distinct values at one fixed digit."""
    n_items, n_digit = codes.shape
    codebook_size = centroids.shape[1]
    other_digits = tuple(d for d in range(n_digit) if d != repair_digit)

    prefix_groups: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_idx, row in enumerate(codes):
        prefix_groups[tuple(int(row[d]) for d in other_digits)].append(item_idx)

    repaired = codes.copy()
    groups_solved = 0
    max_prefix_group = 1
    native_cost = 0.0
    repaired_cost = 0.0

    digit_vectors = pq_subvectors[:, repair_digit, :]
    digit_centroids = centroids[repair_digit]

    for item_indices in prefix_groups.values():
        max_prefix_group = max(max_prefix_group, len(item_indices))
        if len(item_indices) <= 1:
            continue

        current = codes[item_indices, repair_digit]
        if len(np.unique(current)) == len(item_indices):
            continue
        if len(item_indices) > codebook_size:
            raise ValueError(
                f"prefix group size {len(item_indices)} exceeds codebook capacity "
                f"{codebook_size} at repair digit {repair_digit}"
            )

        groups_solved += 1
        vectors = digit_vectors[item_indices]
        costs = ((vectors[:, None, :] - digit_centroids[None, :, :]) ** 2).sum(axis=-1)
        rows, assigned_codes = linear_sum_assignment(costs)
        assignment = np.empty(len(item_indices), dtype=np.int64)
        assignment[rows] = assigned_codes

        native_cost += float(costs[np.arange(len(item_indices)), current].sum())
        repaired_cost += float(costs[np.arange(len(item_indices)), assignment].sum())
        repaired[item_indices, repair_digit] = assignment

    before = collision_stats(codes)
    after = collision_stats(repaired)
    if after["duplicate_excess"] != 0:
        raise RuntimeError(
            f"digit-{repair_digit} assignment left {after['duplicate_excess']} duplicate codes"
        )

    report = {
        "repair_digit": int(repair_digit),
        "groups_solved": int(groups_solved),
        "max_other_digit_group_size": int(max_prefix_group),
        "changed_items": int(np.any(repaired != codes, axis=1).sum()),
        "native_group_distortion": native_cost,
        "repaired_group_distortion": repaired_cost,
        "distortion_increase": float(repaired_cost - native_cost),
        "before": before,
        "after": after,
    }
    return repaired, report


def repair_code_digit(
    codes: np.ndarray,
    digit_vectors: np.ndarray,
    digit_centroids: np.ndarray,
    repair_digit: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Make hybrid codes injective by reassigning one metric-backed digit.

    Hybrid tokenizers such as RQ-OPQ do not have equal-width subspaces for
    every coordinate: RQ centroids live in the full embedding space while an
    OPQ digit lives in one rotated subspace.  This wrapper exposes the same
    minimum-cost assignment for one OPQ coordinate without fabricating metric
    tensors for the RQ positions.
    """
    codes = np.asarray(codes, dtype=np.int64)
    digit_vectors = np.asarray(digit_vectors, dtype=np.float32)
    digit_centroids = np.asarray(digit_centroids, dtype=np.float32)
    repair_digit = int(repair_digit)
    if codes.ndim != 2:
        raise ValueError(f"codes must be rank-2, got {codes.shape}")
    if not 0 <= repair_digit < codes.shape[1]:
        raise ValueError(
            f"repair_digit={repair_digit} outside [0,{codes.shape[1]})"
        )
    if digit_vectors.ndim != 2 or digit_vectors.shape[0] != codes.shape[0]:
        raise ValueError(
            "digit_vectors must have shape [n_items,subvector_dim], got "
            f"{digit_vectors.shape}"
        )
    if digit_centroids.ndim != 2:
        raise ValueError(
            "digit_centroids must have shape [codebook_size,subvector_dim]"
        )
    if digit_vectors.shape[1] != digit_centroids.shape[1]:
        raise ValueError(
            f"vector/centroid dimensions differ: {digit_vectors.shape[1]} vs "
            f"{digit_centroids.shape[1]}"
        )

    # _repair_one_digit only indexes the requested slice. Compact dummy
    # tensors let the hybrid wrapper reuse the same tested assignment solver.
    vectors = np.zeros(
        (codes.shape[0], codes.shape[1], digit_vectors.shape[1]),
        dtype=np.float32,
    )
    centroids = np.zeros(
        (codes.shape[1], digit_centroids.shape[0], digit_centroids.shape[1]),
        dtype=np.float32,
    )
    vectors[:, repair_digit] = digit_vectors
    centroids[repair_digit] = digit_centroids
    repaired, selected = _repair_one_digit(
        codes, vectors, centroids, repair_digit
    )
    return repaired, {
        "strategy": "hungarian_fixed_digit",
        "selected_repair_digit": repair_digit,
        "selected": selected,
    }


def repair_product_codes(
    codes: np.ndarray,
    pq_subvectors: np.ndarray,
    centroids: np.ndarray,
    repair_digit: int | str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Produce injective product codes with minimum one-coordinate distortion.

    For ``repair_digit='auto'``, all digit choices are solved independently and
    the globally least-distorting solution is returned.  Code positions remain
    fixed; only one position is allowed to change for the whole catalog.
    """
    codes = np.asarray(codes, dtype=np.int64)
    pq_subvectors = np.asarray(pq_subvectors, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    if pq_subvectors.shape[:2] != codes.shape:
        raise ValueError(
            f"subvector/code shape mismatch: {pq_subvectors.shape[:2]} vs {codes.shape}"
        )
    if centroids.shape[0] != codes.shape[1]:
        raise ValueError(
            f"centroid/code digit mismatch: {centroids.shape[0]} vs {codes.shape[1]}"
        )

    if repair_digit == "auto":
        candidates = range(codes.shape[1])
    else:
        digit = int(repair_digit)
        if not 0 <= digit < codes.shape[1]:
            raise ValueError(f"repair_digit={digit} outside [0, {codes.shape[1]})")
        candidates = [digit]

    solved = []
    for digit in candidates:
        repaired, report = _repair_one_digit(
            codes=codes,
            pq_subvectors=pq_subvectors,
            centroids=centroids,
            repair_digit=digit,
        )
        solved.append((repaired, report))

    best_codes, best_report = min(
        solved,
        key=lambda pair: (pair[1]["distortion_increase"], pair[1]["changed_items"]),
    )
    report = {
        "strategy": "hungarian",
        "selected_repair_digit": best_report["repair_digit"],
        "selected": best_report,
        "candidate_summaries": [entry[1] for entry in solved],
    }
    return best_codes, report
