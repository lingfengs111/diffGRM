#!/usr/bin/env python3
"""Prepare and audit a CleanGR-backed DiffGRM catalog before training."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer
from genrec.utils import get_config, get_dataset


LEGACY_EXPECTED_SPLITS = {
    ("AmazonReviews2014CleanGR", "Beauty"): (131413, 22363, 22363, 12101),
    ("AmazonReviews2023CleanGR", "Video_Games"): (530300, 94762, 94762, 25612),
}


def expected_protocol(config: dict, dataset_name: str) -> tuple[int, int, int, int]:
    configured = config.get("expected_protocol")
    if configured:
        required = ("train", "val", "test", "items")
        missing = [key for key in required if key not in configured]
        if missing:
            raise ValueError(f"expected_protocol is missing keys: {missing}")
        return tuple(int(configured[key]) for key in required)

    legacy = LEGACY_EXPECTED_SPLITS.get((dataset_name, config.get("category")))
    if legacy is not None:
        return legacy

    stats_path = Path(config["data_dir"]) / "stats.json"
    if not stats_path.is_file():
        raise ValueError(
            "No expected_protocol in config and no prepared stats.json at "
            f"{stats_path}"
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return (
        int(stats["splits"]["train_rows"]),
        int(stats["splits"]["valid_rows"]),
        int(stats["splits"]["test_rows"]),
        int(stats["core"]["items"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Validate prepared rows/catalog without generating embeddings or SIDs.",
    )
    args = parser.parse_args()

    config = get_config("DIFF_GRM", args.dataset, args.config, {})
    config["device"] = "cuda"
    config["use_ddp"] = False
    config["accelerator"] = Accelerator()

    dataset = get_dataset(args.dataset)(config)
    splits = dataset.split()
    actual = (
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
        dataset.n_items - 1,
    )
    model_max_history = int(config.get("max_history_len", 20))
    prepared_max_history = getattr(dataset, "prepared_max_history_items", None)
    expected = expected_protocol(config, args.dataset)
    if actual != expected:
        raise RuntimeError(f"protocol mismatch: expected={expected}, actual={actual}")

    if args.data_only:
        print(
            f"DATA_AUDIT_OK dataset={args.dataset} category={config.get('category')} "
            f"train={actual[0]} val={actual[1]} test={actual[2]} items={actual[3]} "
            f"max_history_len={model_max_history} "
            f"prepared_cap={prepared_max_history}"
        )
        return

    tokenizer = DIFF_GRMTokenizer(config, dataset)
    token_rows = np.asarray(
        [tokenizer.item2tokens[item] for item in dataset.id_mapping["id2item"][1:]],
        dtype=np.int64,
    )
    offsets = tokenizer.sid_offset + np.arange(
        int(config["n_digit"]), dtype=np.int64
    ) * int(config["codebook_size"])
    codes = token_rows - offsets[None, :]
    if codes.min() < 0 or codes.max() >= int(config["codebook_size"]):
        raise RuntimeError(
            f"invalid raw code range: min={codes.min()}, max={codes.max()}"
        )
    unique = len(np.unique(codes, axis=0))
    if unique != len(codes):
        raise RuntimeError(f"catalog collision: unique={unique}, items={len(codes)}")

    digest = hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()
    print(
        f"AUDIT_OK dataset={args.dataset} train={actual[0]} val={actual[1]} "
        f"test={actual[2]} items={actual[3]} unique={unique} sha256={digest} "
        f"max_history_len={model_max_history} prepared_cap={prepared_max_history}"
    )


if __name__ == "__main__":
    main()
