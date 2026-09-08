from types import SimpleNamespace

import torch

from genrec.models.DIFF_GRM.catalog_decode import (
    CatalogConstraint,
    catalog_order_marginal_decode,
    catalog_uncertainty_decode,
)


class _Tokenizer:
    sid_offset = 2

    def __init__(self):
        raw = {
            "a": (0, 0, 0),
            "b": (0, 1, 1),
            "c": (1, 0, 1),
            "d": (1, 1, 0),
        }
        self.item2tokens = {
            item: tuple(code + self.sid_offset + d * 3 for d, code in enumerate(row))
            for item, row in raw.items()
        }


class _Model:
    n_digit = 3
    codebook_size = 3

    def __init__(self):
        self.config = {
            "current_split": "val",
            "vectorized_beam_search": {
                "top_k_final": 3,
                "val": {"beam_act": 8, "beam_max": 8},
            },
            "catalog_beam": {"premerge_factor": None},
        }

    def forward_decoder_only(self, batch, digit=None, use_cache=False):
        state = batch["decoder_input_ids"]
        mask = batch["mask_positions"]
        batch_size = state.shape[0]
        logits = torch.zeros(batch_size, self.n_digit, self.codebook_size)
        # Code 2 is deliberately strongest everywhere, but no catalog item
        # contains it. The constraint must prevent it from being emitted.
        logits[..., 2] = 9.0
        logits[..., 1] = 1.0
        logits = logits.masked_fill(~mask[..., None], -20.0)
        return SimpleNamespace(logits=logits)


def test_constraint_indexes_arbitrary_partial_states():
    index = CatalogConstraint([[0, 0, 0], [0, 1, 1], [1, 0, 1]], 3)
    assert index.allowed([0, -1, -1]).tolist() == [3, 4, 6, 7]
    assert index.allowed([-1, 0, 1]).tolist() == [1]
    assert index.allowed([2, -1, -1]) is None


def test_catalog_decode_only_returns_unique_legal_sequences():
    model = _Model()
    tokenizer = _Tokenizer()
    result = catalog_order_marginal_decode(
        model,
        encoder_hidden=torch.zeros(2, 4, 5),
        tokenizer=tokenizer,
        n_return_sequences=3,
    )
    assert result.shape == (2, 3, 3)
    legal = {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)}
    for per_user in result.tolist():
        rows = [tuple(row) for row in per_user]
        assert len(set(rows)) == len(rows)
        assert set(rows).issubset(legal)


def test_uncertainty_decode_returns_scored_legal_sequences():
    model = _Model()
    tokenizer = _Tokenizer()
    result, scores = catalog_uncertainty_decode(
        model,
        encoder_hidden=torch.zeros(2, 4, 5),
        tokenizer=tokenizer,
        n_return_sequences=3,
        return_scores=True,
    )
    assert result.shape == (2, 3, 3)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()
    legal = {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)}
    for per_user in result.tolist():
        rows = [tuple(row) for row in per_user]
        assert len(set(rows)) == len(rows)
        assert set(rows).issubset(legal)
