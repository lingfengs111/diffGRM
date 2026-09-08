from types import SimpleNamespace

import torch

from genrec.diagnostics import (
    catalog_subset_diagnostics,
    catalog_diagnostics,
    conditional_diagnostics,
    coordinate_subset_masks,
    generation_diagnostics,
    stable_target_ranks,
    subset_digits,
)


def test_generation_diagnostics_locates_first_error():
    labels = torch.tensor([[1, 2, 3], [1, 2, 3]])
    preds = torch.tensor([[[1, 9, 3]], [[1, 2, 3]]])
    metrics = generation_diagnostics(preds, labels)
    assert metrics["diag/free/digit2_acc"].tolist() == [1.0, 1.0]
    assert metrics["diag/free/prefix2_acc"].tolist() == [0.0, 1.0]
    assert metrics["diag/free/first_error_digit1"].tolist() == [1.0, 0.0]
    assert metrics["diag/free/full_sid_acc"].tolist() == [0.0, 1.0]


def test_conditional_diagnostics_are_per_digit():
    logits = torch.tensor([[[0.0, 3.0], [2.0, 0.0]]])
    labels = torch.tensor([[1, 1]])
    metrics = conditional_diagnostics(logits, labels)
    assert metrics["diag/conditional/digit0_acc"].item() == 1.0
    assert metrics["diag/conditional/digit1_acc"].item() == 0.0
    assert metrics["diag/conditional/digit0_mrr"].item() == 1.0
    assert metrics["diag/conditional/digit1_mrr"].item() == 0.5


def test_catalog_diagnostics_exposes_hierarchy_and_collisions():
    tokenizer = SimpleNamespace(
        dataset=SimpleNamespace(item2id={"a": 1, "b": 2, "c": 3, "d": 4}),
        item2tokens={
            "a": (3, 7),
            "b": (3, 8),
            "c": (4, 7),
            "d": (4, 7),
        },
        sid_offset=3,
        n_digit=2,
    )
    metrics = catalog_diagnostics(tokenizer, codebook_size=4)
    assert metrics["diag/catalog/unique_sid_ratio"] == 0.75
    assert metrics["diag/catalog/colliding_item_ratio"] == 0.5
    assert metrics["diag/catalog/prefix1_unique_ratio"] == 0.5
    assert metrics["diag/catalog/prefix2_unique_ratio"] == 0.75


def test_coordinate_subsets_are_cardinality_ordered():
    masks = coordinate_subset_masks(3, include_empty=True, include_full=True)
    assert masks == [0, 1, 2, 4, 3, 5, 6, 7]
    assert subset_digits(5, 3) == [0, 2]


def test_stable_target_ranks_use_catalog_order_for_ties():
    scores = torch.tensor([[3.0, 2.0, 2.0], [1.0, 4.0, 0.0]])
    targets = torch.tensor([2, 1])
    assert stable_target_ranks(scores, targets).tolist() == [3, 1]


def test_catalog_subset_diagnostics_are_not_prefix_specific():
    codes = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]).numpy()
    metrics = catalog_subset_diagnostics(codes)
    assert metrics["full_unique_ratio"] == 1.0
    assert metrics["subsets"]["0"]["mean_candidate_count_per_item"] == 2.0
    assert metrics["subsets"]["1"]["mean_candidate_count_per_item"] == 2.0
    assert metrics["subsets"]["0-1"]["singleton_item_rate"] == 1.0
    assert abs(metrics["pairwise_normalized_mutual_information"]["0-1"]) < 1e-12
