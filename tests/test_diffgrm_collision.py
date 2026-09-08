import numpy as np

from genrec.models.DIFF_GRM.collision import (
    append_dedup_digit,
    collision_stats,
    repair_code_digit,
    repair_product_codes,
)


def test_append_dedup_digit_is_injective() -> None:
    native = np.asarray([[0, 1], [0, 1], [0, 2], [3, 3]], dtype=np.int64)
    repaired, report = append_dedup_digit(native, codebook_size=4)

    assert repaired.tolist() == [[0, 1, 0], [0, 1, 1], [0, 2, 0], [3, 3, 0]]
    assert report["before"]["duplicate_excess"] == 1
    assert report["after"]["duplicate_excess"] == 0


def test_hungarian_product_repair_is_injective() -> None:
    native = np.asarray([[0, 1], [0, 1], [0, 2], [3, 3]], dtype=np.int64)
    # Two 1-D PQ subspaces with four centroids each.
    centroids = np.asarray(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[0.0], [1.0], [2.0], [3.0]],
        ],
        dtype=np.float32,
    )
    subvectors = np.asarray(
        [
            [[0.05], [1.05]],
            [[0.15], [1.10]],
            [[0.05], [2.05]],
            [[3.00], [3.00]],
        ],
        dtype=np.float32,
    )

    repaired, report = repair_product_codes(
        native,
        pq_subvectors=subvectors,
        centroids=centroids,
        repair_digit="auto",
    )

    assert collision_stats(repaired)["duplicate_excess"] == 0
    assert report["selected"]["changed_items"] >= 1
    assert report["selected_repair_digit"] in (0, 1)


def test_fixed_digit_repair_supports_hybrid_code_geometry() -> None:
    native = np.asarray(
        [[3, 2, 0, 1], [3, 2, 0, 1], [3, 2, 2, 1], [1, 0, 3, 3]],
        dtype=np.int64,
    )
    vectors = np.asarray([[1.05], [1.15], [1.05], [3.0]], dtype=np.float32)
    centroids = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    repaired, report = repair_code_digit(
        native, vectors, centroids, repair_digit=3
    )
    assert collision_stats(repaired)["duplicate_excess"] == 0
    assert np.array_equal(repaired[:, :3], native[:, :3])
    assert report["selected_repair_digit"] == 3
