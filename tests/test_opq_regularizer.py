import numpy as np
import torch

from genrec.models.DIFF_GRM.opq_regularizer import (
    hard_product_codes,
    regularize_product_centroids,
    regularized_pq_loss,
)


def test_regularized_pq_loss_is_finite_and_differentiable():
    torch.manual_seed(7)
    inputs = torch.randn(32, 3, 4)
    initial = torch.randn(3, 8, 4)
    centroids = initial.clone().requires_grad_(True)
    loss, metrics = regularized_pq_loss(
        inputs,
        centroids,
        initial,
        temperature=0.2,
        balance_weight=0.1,
        mi_weight=0.1,
        hardness_weight=0.01,
        anchor_weight=0.01,
        straight_through=False,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(centroids.grad).all()
    assert set(metrics) == {
        "loss", "reconstruction", "balance_kl", "soft_pairwise_mi",
        "assignment_entropy", "centroid_anchor"
    }


def test_regularizer_returns_valid_hard_codes():
    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(64, 2, 3)).astype(np.float32)
    centroids = rng.normal(size=(2, 8, 3)).astype(np.float32)
    codes, learned, report = regularize_product_centroids(
        inputs,
        centroids,
        np.ones(64, dtype=bool),
        steps=3,
        batch_size=32,
        learning_rate=1e-3,
        temperature=0.2,
        balance_weight=0.1,
        mi_weight=0.1,
        hardness_weight=0.01,
        anchor_weight=0.01,
        device="cpu",
        seed=13,
        straight_through=True,
    )
    assert codes.shape == (64, 2)
    assert learned.shape == centroids.shape
    assert codes.min() >= 0 and codes.max() < 8
    np.testing.assert_array_equal(codes, hard_product_codes(inputs, learned))
    assert report["history"][-1]["step"] == 3
