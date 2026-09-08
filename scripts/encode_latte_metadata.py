#!/usr/bin/env python3
"""Encode the exact metadata sentence view used by Latte/PSID."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def clean_text(value) -> str:
    if isinstance(value, list):
        value = ", ".join(map(str, value))
    elif value is None:
        value = ""
    else:
        value = str(value)
    value = html.unescape(value).strip()
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[\n\t]", " ", value)
    value = re.sub(r" +", " ", value)
    return re.sub(r"[^\x00-\x7F]", " ", value)


def feature_text(value) -> str:
    # This mirrors Latte's AmazonReviews2023._feature_process, including its
    # punctuation between list-valued metadata fields.
    if isinstance(value, float):
        return f"{value}. "
    if isinstance(value, list) and value:
        return ", ".join(clean_text(part) for part in value) + ". "
    return clean_text(value) + " "


def latte_sentence(row: dict) -> str:
    return "".join(
        feature_text(row.get(field))
        for field in ("title", "features", "categories", "description")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-vocab", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-embeddings", type=Path, required=True)
    parser.add_argument("--output-texts", type=Path, required=True)
    parser.add_argument(
        "--model", default="sentence-transformers/sentence-t5-base"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.output_embeddings, args.output_texts)
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("output exists; pass --force to replace both outputs")

    with args.item_vocab.open(newline="", encoding="utf-8") as handle:
        items = [row["item_id"].strip() for row in csv.DictReader(handle)]
    wanted = set(items)
    metadata = {}
    opener = gzip.open if args.metadata.suffix == ".gz" else open
    with opener(args.metadata, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            item = str(row.get("parent_asin") or row.get("asin") or "")
            if item in wanted:
                metadata[item] = row
    missing = [item for item in items if item not in metadata]
    if missing:
        raise ValueError(f"metadata missing for {len(missing)} items: {missing[:5]}")

    texts = [latte_sentence(metadata[item]) for item in items]
    args.output_texts.parent.mkdir(parents=True, exist_ok=True)
    with args.output_texts.open("w", encoding="utf-8") as handle:
        for item, sentence in zip(items, texts):
            handle.write(json.dumps({"item_id": item, "text": sentence}) + "\n")

    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
        device=args.device,
    ).astype(np.float32, copy=False)
    args.output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    embeddings.tofile(args.output_embeddings)
    digest = hashlib.sha256()
    for sentence in texts:
        digest.update(sentence.encode("utf-8"))
        digest.update(b"\0")
    diagnostics = {
        "catalog_items": len(items),
        "embedding_shape": list(embeddings.shape),
        "model": args.model,
        "metadata_fields": ["title", "features", "categories", "description"],
        "ordered_text_sha256": digest.hexdigest(),
        "device": args.device,
    }
    args.output_embeddings.with_suffix(
        args.output_embeddings.suffix + ".diagnostics.json"
    ).write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
