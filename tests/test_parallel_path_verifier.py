import torch

from genrec.models.DIFF_GRM.path_verifier import (
    IndependentSIDHistoryEncoder,
    ParallelPathVerifier,
)


def tiny_verifier(coordinate_mode='bidirectional', history_pooling='mean'):
    return ParallelPathVerifier(
        n_digit=4,
        codebook_size=8,
        context_dim=16,
        hidden_dim=32,
        n_head=4,
        coordinate_layers=1,
        set_layers=1,
        dropout=0.0,
        coordinate_mode=coordinate_mode,
        history_pooling=history_pooling,
    ).eval()


def test_parallel_path_verifier_shape_and_gradients():
    torch.manual_seed(3)
    verifier = tiny_verifier().train()
    history = torch.randn(2, 5, 16)
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    candidates = torch.randint(0, 8, (2, 7, 4))
    proposal = torch.randn(2, 7)
    scores = verifier(history, candidates, proposal, mask)
    assert scores.shape == (2, 7)
    scores.square().mean().backward()
    assert verifier.code_embeddings.grad is not None
    assert verifier.code_embeddings.grad.abs().sum() > 0


def test_parallel_path_verifier_is_candidate_permutation_equivariant():
    torch.manual_seed(5)
    verifier = tiny_verifier()
    history = torch.randn(2, 5, 16)
    candidates = torch.randint(0, 8, (2, 7, 4))
    proposal = torch.randn(2, 7)
    permutation = torch.tensor([5, 0, 3, 1, 6, 2, 4])
    with torch.no_grad():
        reference = verifier(history, candidates, proposal)
        permuted = verifier(
            history,
            candidates[:, permutation],
            proposal[:, permutation],
        )
    torch.testing.assert_close(
        reference[:, permutation], permuted, atol=1e-6, rtol=1e-6
    )


def test_all_coordinate_modes_score_and_backpropagate():
    torch.manual_seed(7)
    history = torch.randn(2, 5, 16)
    history_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    candidates = torch.randint(0, 8, (2, 9, 4))
    proposal = torch.randn(2, 9)
    parameter_counts = {}
    for mode in ('bidirectional', 'causal', 'mlp'):
        verifier = tiny_verifier(mode, history_pooling='last').train()
        scores = verifier(history, candidates, proposal, history_mask)
        assert scores.shape == (2, 9)
        scores.mean().backward()
        assert verifier.code_embeddings.grad is not None
        parameter_counts[mode] = sum(p.numel() for p in verifier.parameters())
    assert parameter_counts['causal'] == parameter_counts['bidirectional']
    # The MLP control follows the same leading-order H^2 budget as one
    # Transformer layer; small norm/bias differences are intentionally allowed.
    relative_gap = abs(
        parameter_counts['mlp'] - parameter_counts['bidirectional']
    ) / parameter_counts['bidirectional']
    assert relative_gap < 0.05


def test_independent_history_encoder_masks_padding_and_backpropagates():
    torch.manual_seed(11)
    encoder = IndependentSIDHistoryEncoder(
        n_digit=4,
        codebook_size=8,
        hidden_dim=16,
        n_head=4,
        layers=2,
        max_history_len=5,
        dropout=0.0,
        inner_dim=32,
    ).train()
    history = torch.randint(0, 8, (2, 5, 4))
    history[0, 3:] = -1
    hidden = encoder(history)
    assert hidden.shape == (2, 5, 16)
    assert torch.equal(hidden[0, 3:], torch.zeros_like(hidden[0, 3:]))
    hidden.square().mean().backward()
    assert encoder.code_embeddings.grad is not None
    assert encoder.code_embeddings.grad.abs().sum() > 0
