#!/usr/bin/env python3
"""Build collision-free, semantic-neighbor PIT-lite aliases for an OPQ catalog.

Each item keeps its canonical SID and receives a small number of exclusive
one-coordinate mutations.  The first coordinate is preferred because the
current OPQ experiments diagnose the first decision as the main routing
bottleneck.  Nearest text-embedding neighbours propose replacement codes;
global reservation prevents canonical/canonical, canonical/alias, and
alias/alias collisions.
"""

import argparse
import json
import os
from collections import Counter

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical-sids', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--aliases-per-item', type=int, default=2)
    parser.add_argument('--neighbor-pool', type=int, default=128)
    parser.add_argument('--codebook-size', type=int, default=256)
    parser.add_argument('--preferred-digit', type=int, default=0)
    return parser.parse_args()


def load_inputs(args):
    with open(args.canonical_sids, encoding='utf-8') as handle:
        canonical = json.load(handle)
    items = list(canonical)
    codes = np.asarray([canonical[item] for item in items], dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f'Expected a rectangular SID matrix, got {codes.shape}')
    if len(np.unique(codes, axis=0)) != len(codes):
        raise ValueError('Canonical SID catalog is not collision free')
    if codes.min() < 0 or codes.max() >= args.codebook_size:
        raise ValueError('Canonical SID contains an out-of-range code')

    n_float = os.path.getsize(args.embeddings) // np.dtype(np.float32).itemsize
    if n_float % len(items):
        raise ValueError(
            f'Embedding file has {n_float} floats for {len(items)} catalog items'
        )
    dim = n_float // len(items)
    embeddings = np.fromfile(args.embeddings, dtype=np.float32).reshape(len(items), dim)
    faiss.normalize_L2(embeddings)
    return items, codes, embeddings


def nearest_neighbors(embeddings, pool):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index.search(embeddings, min(pool + 1, len(embeddings)))


def build_aliases(items, codes, distances, neighbors, args):
    if args.aliases_per_item < 1:
        raise ValueError('aliases-per-item includes canonical and must be >= 1')
    n_digit = codes.shape[1]
    if not 0 <= args.preferred_digit < n_digit:
        raise ValueError('preferred-digit is outside the SID width')

    digit_order = [args.preferred_digit] + [
        digit for digit in range(n_digit) if digit != args.preferred_digit
    ]
    reserved = {tuple(row.tolist()) for row in codes}
    item_to_paths = {}
    source_similarities = []
    selected_digits = Counter()

    for item_idx, item in enumerate(items):
        canonical = tuple(codes[item_idx].tolist())
        paths = [canonical]
        for alias_number in range(1, args.aliases_per_item):
            selected = None
            selected_similarity = None
            selected_digit = None

            # Prefer an alternate first route proposed by a semantically close
            # item. Only one coordinate changes, preserving most OPQ content.
            for digit in digit_order:
                for rank in range(1, neighbors.shape[1]):
                    neighbor_idx = int(neighbors[item_idx, rank])
                    replacement = int(codes[neighbor_idx, digit])
                    if replacement == canonical[digit]:
                        continue
                    candidate = list(canonical)
                    candidate[digit] = replacement
                    candidate = tuple(candidate)
                    if candidate in reserved:
                        continue
                    selected = candidate
                    selected_similarity = float(distances[item_idx, rank])
                    selected_digit = digit
                    break
                if selected is not None:
                    break

            # Deterministic exhaustive fallback. This is rarely needed, but
            # makes catalog construction total even for dense local buckets.
            if selected is None:
                for digit in digit_order:
                    start = (
                        canonical[digit] + item_idx + alias_number
                    ) % args.codebook_size
                    for offset in range(args.codebook_size):
                        replacement = (start + offset) % args.codebook_size
                        if replacement == canonical[digit]:
                            continue
                        candidate = list(canonical)
                        candidate[digit] = replacement
                        candidate = tuple(candidate)
                        if candidate not in reserved:
                            selected = candidate
                            selected_similarity = None
                            selected_digit = digit
                            break
                    if selected is not None:
                        break
            if selected is None:
                raise RuntimeError(
                    f'Could not allocate alias {alias_number} for item {item!r}'
                )

            reserved.add(selected)
            paths.append(selected)
            selected_digits[selected_digit] += 1
            if selected_similarity is not None:
                source_similarities.append(selected_similarity)
        item_to_paths[item] = [list(path) for path in paths]

    all_paths = [tuple(path) for paths in item_to_paths.values() for path in paths]
    if len(all_paths) != len(set(all_paths)):
        raise AssertionError('Internal error: output alias catalog contains collisions')
    report = {
        'method': 'pit_lite_semantic_neighbor_one_coordinate_alias',
        'n_items': len(items),
        'n_digit': n_digit,
        'codebook_size': args.codebook_size,
        'paths_per_item': args.aliases_per_item,
        'n_total_paths': len(all_paths),
        'n_unique_paths': len(set(all_paths)),
        'n_collisions': 0,
        'preferred_digit': args.preferred_digit,
        'selected_digit_counts': {
            str(digit): int(selected_digits[digit]) for digit in range(n_digit)
        },
        'mean_source_cosine': (
            float(np.mean(source_similarities)) if source_similarities else None
        ),
        'canonical_sids': os.path.abspath(args.canonical_sids),
        'embeddings': os.path.abspath(args.embeddings),
    }
    return item_to_paths, report


def main():
    args = parse_args()
    items, codes, embeddings = load_inputs(args)
    distances, neighbors = nearest_neighbors(embeddings, args.neighbor_pool)
    item_to_paths, report = build_aliases(
        items, codes, distances, neighbors, args
    )
    payload = {'metadata': report, 'item_to_paths': item_to_paths}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
