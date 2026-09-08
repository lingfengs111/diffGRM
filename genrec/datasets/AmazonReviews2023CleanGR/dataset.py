"""Adapter for Amazon Reviews 2023 datasets prepared by CleanGR.

The adapter deliberately consumes CleanGR's existing item vocabulary, metadata,
and train/valid/test JSONL files.  This keeps DiffGRM and CleanGR on the same
catalog and leave-one-out targets instead of rebuilding a subtly different
dataset inside each repository.
"""

import csv
import json
import os
from pathlib import Path

from datasets import Dataset

from genrec.dataset import AbstractDataset


class AmazonReviews2023CleanGR(AbstractDataset):
    """Load a CleanGR-prepared Amazon Reviews sequential dataset."""

    amazon_release = "2023"
    recommended_max_history_len = 20

    def __init__(self, config: dict):
        super().__init__(config)
        self.category = config.get("category", "Video_Games")
        self.data_dir = os.path.abspath(os.path.expanduser(config["data_dir"]))
        self.item_vocab_file = str(config.get("item_vocab_file", "item_vocab.csv"))
        self.item_texts_file = str(config.get("item_texts_file", "item_texts.csv"))
        self.splits_dir = str(config.get("splits_dir", "splits"))
        self._check_history_protocol(config)
        if self.item_texts_file != "item_texts.csv":
            config.setdefault(
                "metadata_cache_tag",
                Path(self.item_texts_file).stem,
            )
        self.cache_dir = os.path.join(
            config["cache_dir"], self.__class__.__name__, self.category
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.log(
            f"[DATASET] Amazon Reviews {self.amazon_release} CleanGR adapter for "
            f"{self.category}: {self.data_dir}; item_texts={self.item_texts_file}"
        )
        self._load_item_mapping()
        self._load_metadata()
        self._load_prepared_splits()

    def _check_history_protocol(self, config: dict) -> None:
        """Make the shared Amazon max-history protocol visible and safe."""
        standard = int(
            config.get(
                "amazon_standard_max_history_len",
                self.recommended_max_history_len,
            )
        )
        configured = int(config.get("max_history_len", standard))
        if configured != standard:
            acknowledged = bool(
                config.get("allow_nonstandard_amazon_maxlen", False)
            )
            self.log(
                "[AMAZON PROTOCOL] max_history_len="
                f"{configured}; the default comparable protocol is "
                f"{standard}. "
                "Use a run-id suffix such as _l50 for intentional overrides.",
                level="info" if acknowledged else "warning",
            )

        stats_path = Path(self.data_dir) / "stats.json"
        self.prepared_max_history_items = None
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            prepared_cap = stats.get("splits", {}).get("max_history_items")
            if prepared_cap is not None:
                self.prepared_max_history_items = int(prepared_cap)
                if configured > self.prepared_max_history_items:
                    raise ValueError(
                        f"max_history_len={configured} exceeds the prepared history "
                        f"cap={self.prepared_max_history_items} in {stats_path}; "
                        "rebuild the splits with a larger cap first"
                    )

    def _required_path(self, *parts: str) -> str:
        path = os.path.join(self.data_dir, *parts)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required CleanGR data file not found: {path}")
        return path

    def _load_item_mapping(self) -> None:
        vocab_path = self._required_path(self.item_vocab_file)
        with open(vocab_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "item_id" not in reader.fieldnames:
                raise ValueError(f"item_vocab.csv lacks item_id column: {vocab_path}")
            for row in reader:
                item = row["item_id"].strip()
                if not item or item in self.id_mapping["item2id"]:
                    continue
                self.id_mapping["item2id"][item] = len(
                    self.id_mapping["id2item"]
                )
                self.id_mapping["id2item"].append(item)

    def _load_metadata(self) -> None:
        metadata_path = self._required_path(self.item_texts_file)
        item2meta = {}
        with open(metadata_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"item_id", "text"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"item text file lacks required columns {sorted(required)}: "
                    f"{metadata_path}"
                )
            for row in reader:
                item = row["item_id"].strip()
                if item not in self.item2id:
                    continue
                text = (row.get("text") or "").strip()
                if not text:
                    text = (row.get("title") or item).strip()
                item2meta[item] = text

        missing = [item for item in self.id_mapping["id2item"][1:] if item not in item2meta]
        if missing:
            self.log(
                f"[DATASET] Missing metadata for {len(missing)} items; using raw item IDs",
                level="warning",
            )
            for item in missing:
                item2meta[item] = item
        self.item2meta = item2meta

    def _register_user(self, user: str) -> None:
        if user not in self.id_mapping["user2id"]:
            self.id_mapping["user2id"][user] = len(self.id_mapping["id2user"])
            self.id_mapping["id2user"].append(user)

    def _load_jsonl_split(self, filename: str) -> Dataset:
        path = self._required_path(self.splits_dir, filename)
        users = []
        item_seqs = []
        unknown_items = set()

        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                user = str(row["user_id"])
                history = row.get("history_item_ids", row.get("inter_history", []))
                target = str(row["target_id"])
                seq = [str(item) for item in history] + [target]

                for item in seq:
                    if item not in self.item2id:
                        unknown_items.add(item)
                self._register_user(user)
                users.append(user)
                item_seqs.append(seq)

        if unknown_items:
            sample = sorted(unknown_items)[:10]
            raise ValueError(
                f"{filename} references {len(unknown_items)} items absent from "
                f"item_vocab.csv; sample={sample}"
            )
        return Dataset.from_dict({"user": users, "item_seq": item_seqs})

    def _load_prepared_splits(self) -> None:
        # Load test first: it has one complete, chronologically ordered sequence
        # for every user and is also useful for dataset statistics.
        test = self._load_jsonl_split("test.jsonl")
        valid = self._load_jsonl_split("valid.jsonl")
        train = self._load_jsonl_split("train.jsonl")
        self.split_data = {"train": train, "val": valid, "test": test}
        self.all_item_seqs = {
            user: seq for user, seq in zip(test["user"], test["item_seq"])
        }

        self.log(
            f"[DATASET] Prepared rows: train={len(train)}, "
            f"val={len(valid)}, test={len(test)}"
        )

    def _download_and_process_raw(self):
        raise RuntimeError(
            "AmazonReviews2023CleanGR only reads an existing CleanGR prepared dataset"
        )
