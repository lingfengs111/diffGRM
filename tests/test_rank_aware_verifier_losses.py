import torch

from scripts.train_candidate_aware_verifier import (
    duplicate_prefix_mask,
    ndcg_lambda_dpo_loss,
    prefix_pairwise_loss,
)


def test_duplicate_prefix_mask_keeps_first_occurrence_only():
    candidates = torch.tensor([[
        [1, 2, 3],
        [1, 2, 4],
        [1, 5, 6],
        [7, 8, 9],
    ]])
    assert duplicate_prefix_mask(candidates, 1).tolist() == [
        [False, True, True, False]
    ]
    assert duplicate_prefix_mask(candidates, 2).tolist() == [
        [False, True, False, False]
    ]
    assert duplicate_prefix_mask(candidates, 3).tolist() == [
        [False, False, False, False]
    ]


def test_prefix_pairwise_loss_rewards_positive_prefixes():
    candidates = torch.tensor([[
        [1, 2, 3],
        [1, 4, 5],
        [6, 7, 8],
    ]])
    weights = torch.ones(1)
    weak = torch.tensor([[[0.0, 0.0, 0.0],
                          [0.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0]]], requires_grad=True)
    strong = weak.detach().clone()
    strong[:, 0] += 2.0
    weak_loss, _, _ = prefix_pairwise_loss(
        weak, candidates, weights, 1.0, [2, 3, 3]
    )
    strong_loss, _, adaptive = prefix_pairwise_loss(
        strong, candidates, weights, 1.0, [2, 3, 3]
    )
    assert strong_loss < weak_loss
    assert torch.isclose(adaptive.sum(), torch.tensor(1.0))
    weak_loss.backward()
    assert torch.isfinite(weak.grad).all()


def test_reference_anchored_lambda_dpo_rewards_positive_relative_gain():
    proposal = torch.tensor([[0.0, 1.0, 0.5]])
    reference = torch.tensor([[0.0, 1.0, 0.5]])
    unchanged = reference.clone().requires_grad_()
    improved = torch.tensor([[2.0, 1.0, 0.5]], requires_grad=True)
    weights = torch.ones(1)
    unchanged_loss, unchanged_lambda = ndcg_lambda_dpo_loss(
        unchanged, reference, proposal, weights, 0.5, 10, 0.75
    )
    improved_loss, improved_lambda = ndcg_lambda_dpo_loss(
        improved, reference, proposal, weights, 0.5, 10, 0.75
    )
    assert improved_loss < unchanged_loss
    assert torch.isclose(unchanged_lambda.sum(), torch.tensor(1.0))
    assert torch.isclose(improved_lambda.sum(), torch.tensor(1.0))
    unchanged_loss.backward()
    assert torch.isfinite(unchanged.grad).all()
