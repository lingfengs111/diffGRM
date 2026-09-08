import csv
import json

from genrec.datasets.AmazonReviews2023CleanGR.dataset import AmazonReviews2023CleanGR
from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer


class FakeAccelerator:
    is_main_process = True


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_adapter_supports_alternate_item_texts_and_split_directory(tmp_path):
    with (tmp_path / "vocab.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id"])
        writer.writeheader()
        writer.writerows([{"item_id": item} for item in ("a", "b", "c")])
    with (tmp_path / "item_texts_reviews.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "title", "text"])
        writer.writeheader()
        writer.writerows(
            {"item_id": item, "title": item.upper(), "text": f"review text {item}"}
            for item in ("a", "b", "c")
        )
    split_dir = tmp_path / "prepared"
    split_dir.mkdir()
    row = {
        "user_id": "u1",
        "target_id": "c",
        "history_item_ids": ["a", "b"],
    }
    for split in ("train", "valid", "test"):
        write_jsonl(split_dir / f"{split}.jsonl", [row])

    config = {
        "accelerator": FakeAccelerator(),
        "cache_dir": str(tmp_path / "cache"),
        "category": "Musical_Instruments",
        "data_dir": str(tmp_path),
        "item_vocab_file": "vocab.csv",
        "item_texts_file": "item_texts_reviews.csv",
        "splits_dir": "prepared",
    }
    dataset = AmazonReviews2023CleanGR(config)

    assert dataset.n_items == 4
    assert dataset.item2meta["a"] == "review text a"
    assert len(dataset.split()["train"]) == 1
    assert config["metadata_cache_tag"] == "item_texts_reviews"


def test_embedding_cache_basename_includes_optional_metadata_tag():
    tokenizer = DIFF_GRMTokenizer.__new__(DIFF_GRMTokenizer)
    tokenizer.config = {"sent_emb_model": "org/sentence-t5-base"}
    assert tokenizer._embedding_cache_basename() == "sentence-t5-base"

    tokenizer.config["metadata_cache_tag"] = "rich reviews/top5"
    assert (
        tokenizer._embedding_cache_basename()
        == "sentence-t5-base_meta-rich_reviews_top5"
    )
