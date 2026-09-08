import importlib.util
from pathlib import Path

import pytest
import torch


# The lightweight unit-test environment intentionally omits Hugging Face
# ``datasets``.  Load this self-contained module without importing the model
# package, whose __init__ eagerly imports the full data stack.
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'genrec/models/DIFF_GRM/parallel_drafter.py'
)
SPEC = importlib.util.spec_from_file_location('parallel_drafter', MODULE_PATH)
PARALLEL_DRAFTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARALLEL_DRAFTER)
PairwisePathSelector = PARALLEL_DRAFTER.PairwisePathSelector
CausalResidualCorrector = PARALLEL_DRAFTER.CausalResidualCorrector
catalog_unary_scores = PARALLEL_DRAFTER.catalog_unary_scores
code_rows = PARALLEL_DRAFTER.code_rows
candidate_tree_diagnostics = PARALLEL_DRAFTER.candidate_tree_diagnostics
linear_curriculum_weight = PARALLEL_DRAFTER.linear_curriculum_weight

SET_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'genrec/models/DIFF_GRM/set_drafter.py'
)
SET_SPEC = importlib.util.spec_from_file_location('set_drafter', SET_MODULE_PATH)
SET_DRAFTER = importlib.util.module_from_spec(SET_SPEC)
SET_SPEC.loader.exec_module(SET_DRAFTER)
conditional_catalog_scores = SET_DRAFTER.conditional_catalog_scores
masked_standardize = SET_DRAFTER.masked_standardize
sample_typed_subset_masks = SET_DRAFTER.sample_typed_subset_masks
typed_subset_patterns = SET_DRAFTER.typed_subset_patterns


def test_catalog_unary_scores_rank_legal_items():
    logits = torch.tensor(
        [[[3.0, 0.0], [0.0, 3.0]], [[0.0, 3.0], [3.0, 0.0]]]
    )
    catalog = torch.tensor([[0, 1], [1, 0], [0, 0]])
    scores = catalog_unary_scores(logits, catalog)
    assert scores.argmax(dim=1).tolist() == [0, 1]


def test_linear_curriculum_weight_reaches_end_and_then_stays_there():
    weights = [
        linear_curriculum_weight(epoch, 1.0, 0.2, decay_epochs=5)
        for epoch in range(1, 8)
    ]
    assert weights == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match='one-indexed'):
        linear_curriculum_weight(0, 1.0, 0.0, decay_epochs=5)


def test_code_rows_use_collision_free_catalog_order():
    catalog = torch.tensor([[1, 0], [0, 1], [1, 1]])
    targets = torch.tensor([[1, 1], [1, 0], [0, 1]])
    assert code_rows(targets, catalog, codebook_size=2).tolist() == [2, 0, 1]
    with pytest.raises(ValueError, match="absent"):
        code_rows(torch.tensor([[0, 0]]), catalog, codebook_size=2)


def test_code_rows_ignores_non_identity_latent_prefix():
    catalog = torch.tensor([[0, 1, 2], [0, 2, 1], [0, 3, 3]])
    targets = torch.tensor([[7, 2, 1], [4, 1, 2]])
    assert code_rows(
        targets,
        catalog,
        codebook_size=8,
        identity_start_digit=1,
    ).tolist() == [1, 0]


def test_pairwise_selector_is_parallel_and_differentiable():
    selector = PairwisePathSelector(4, 8, context_dim=6, rank=4)
    hidden = torch.randn(3, 4, 6, requires_grad=True)
    catalog = torch.randint(0, 8, (11, 4))
    scores = selector(hidden, catalog)
    assert scores.shape == (3, 11)
    scores.sum().backward()
    assert hidden.grad is not None
    assert selector.left.grad is not None


def test_causal_corrector_is_exact_base_at_initialization():
    corrector = CausalResidualCorrector(
        3, 5, context_dim=7, state_dim=4, rank=3
    )
    hidden = torch.randn(2, 3, 7)
    logits = torch.randn(2, 3, 5)
    candidates = torch.randint(0, 5, (2, 6, 3))
    output = corrector(hidden, logits, candidates)

    expected = torch.zeros(2, 6)
    base_log_probs = torch.log_softmax(logits, dim=-1)
    for digit in range(3):
        expected += base_log_probs[:, digit].gather(
            1, candidates[:, :, digit]
        )
    assert output['path_log_probs'].shape == (2, 6)
    assert torch.allclose(output['path_log_probs'], expected, atol=1e-6)
    assert torch.count_nonzero(output['residual_logits']) == 0


def test_causal_corrector_uses_prefixes_without_future_leakage():
    corrector = CausalResidualCorrector(
        3, 5, context_dim=7, state_dim=4, rank=3
    )
    for projection in corrector.correction_out:
        torch.nn.init.normal_(projection.weight)
    hidden = torch.randn(1, 3, 7, requires_grad=True)
    logits = torch.randn(1, 3, 5)
    # Same first token, different future: corrected level-0 distribution must
    # be identical.  Changing the prefix may affect level 1.
    candidates = torch.tensor([[[1, 2, 3], [1, 4, 0], [2, 2, 3]]])
    output = corrector(hidden, logits, candidates)
    corrected = output['corrected_logits']
    assert torch.allclose(corrected[:, 0, 0], corrected[:, 1, 0])
    assert not torch.allclose(corrected[:, 0, 1], corrected[:, 2, 1])
    (-output['path_log_probs'].mean()).backward()
    assert hidden.grad is not None
    assert corrector.prefix_gru.weight_hh.grad is not None


def test_triple_residual_warm_starts_from_pairwise_and_is_differentiable():
    pairwise = PairwisePathSelector(4, 8, context_dim=6, rank=4)
    selector = PairwisePathSelector(
        4, 8, context_dim=6, rank=4, triple_rank=3
    )
    incompatible = selector.load_state_dict(pairwise.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith('triple_') for key in incompatible.missing_keys)

    hidden = torch.randn(2, 4, 6, requires_grad=True)
    catalog = torch.randint(0, 8, (13, 4))
    with torch.no_grad():
        selector.triple_scale.fill_(0.05)
    scores = selector(hidden, catalog)
    assert scores.shape == (2, 13)
    scores.sum().backward()
    assert selector.triple_factors.grad is not None
    assert selector.triple_gates[0].weight.grad is not None


def test_adaptive_tree_order_finds_more_shared_prefixes():
    # Digit 1 is shared by every candidate, while digit 0 is distinct.  The
    # adaptive order should therefore beat the fixed 0->1->2 ordering.
    candidates = torch.tensor(
        [[[0, 7, 1], [1, 7, 2], [2, 7, 3], [3, 7, 4]]]
    )
    result = candidate_tree_diagnostics(candidates, codebook_size=8)
    assert result['best_order_nodes'].item() < result['fixed_order_nodes'].item()
    best_order = result['orders'][result['best_order_index'].item()]
    assert best_order[0] == 1
    # Choosing an order only rearranges known candidates; recall is unchanged.
    assert result['independent_nodes'].item() == 12


def test_typed_subset_patterns_cover_all_nonempty_masks():
    patterns = typed_subset_patterns(4)
    assert patterns.shape == (15, 4)
    assert patterns.unique(dim=0).shape[0] == 15
    sampled = sample_typed_subset_masks(
        7, 4, n_views=3, min_masked=2, max_masked=3
    )
    assert sampled.shape == (7, 3, 4)
    widths = sampled.sum(dim=-1)
    assert widths.min().item() >= 2
    assert widths.max().item() <= 3


def test_conditional_catalog_scores_respect_observed_coordinates():
    catalog = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    logits = torch.tensor([[[0.0, 0.0], [0.0, 3.0]]])
    observed = torch.tensor([[0, 0]])
    mask = torch.tensor([[False, True]])
    scores, valid = conditional_catalog_scores(
        logits, catalog, observed, mask
    )
    assert valid.tolist() == [[True, True, False, False]]
    assert scores.argmax(dim=1).item() == 1
    normalized = masked_standardize(scores, valid)
    assert torch.isfinite(normalized[valid]).all()
    assert torch.isneginf(normalized[~valid]).all()
