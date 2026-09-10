#!/usr/bin/env python3
"""Destroy item/SID semantics while preserving the exact SID catalog."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    with args.reference.open() as handle:
        reference = json.load(handle)
    if not isinstance(reference, dict) or not reference:
        raise ValueError("reference must be a non-empty item-to-SID dictionary")

    item_ids = list(reference)
    paths = [tuple(int(code) for code in reference[item]) for item in item_ids]
    lengths = {len(path) for path in paths}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent SID lengths: {sorted(lengths)}")
    unique_before = len(set(paths))
    if unique_before != len(paths):
        raise ValueError(
            f"reference is not collision-free: {len(paths) - unique_before} duplicates"
        )

    permutation = np.random.default_rng(args.seed).permutation(len(paths))
    permuted = {
        item: list(paths[int(source_index)])
        for item, source_index in zip(item_ids, permutation)
    }
    unchanged = sum(reference[item] == permuted[item] for item in item_ids)
    if len({tuple(path) for path in permuted.values()}) != len(paths):
        raise AssertionError("permutation unexpectedly introduced collisions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(permuted, handle, separators=(",", ":"))
    metadata = {
        "control": "random item-to-SID permutation",
        "reference": str(args.reference.resolve()),
        "reference_sha256": file_sha256(args.reference),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "seed": args.seed,
        "num_items": len(item_ids),
        "n_digit": lengths.pop(),
        "unique_paths": len(paths),
        "collisions": 0,
        "unchanged_item_assignments": unchanged,
        "preserves_exact_path_multiset": sorted(paths) == sorted(
            tuple(path) for path in permuted.values()
        ),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with meta_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
