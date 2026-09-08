import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'genrec/models/AR_GRM/ann_drafter.py'
)
SPEC = importlib.util.spec_from_file_location('ann_drafter_under_test', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ExactMIPSDrafter = MODULE.ExactMIPSDrafter


class TinyAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_digit = 2
        self.codebook_size = 4
        self.n_embd = 6
        self.tokenizer = SimpleNamespace(sid_offset=1)
        self.embedding = nn.Embedding(9, 6)
        self.item_mlp = nn.Sequential(nn.Linear(12, 6), nn.ReLU(), nn.Linear(6, 6))


def test_exact_mips_shapes_for_both_item_modes():
    ar = TinyAR()
    catalog = torch.tensor([[0, 0], [0, 1], [1, 0], [3, 3]])
    hidden = torch.randn(3, 4, 6)
    history = torch.tensor(
        [
            [[0, 0], [1, 1], [-1, -1], [-1, -1]],
            [[0, 0], [1, 1], [2, 2], [-1, -1]],
            [[0, 0], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    for mode in ('id', 'opq'):
        model = ExactMIPSDrafter(ar, catalog, mode)
        scores = model(hidden, history)
        assert scores.shape == (3, 4)
        assert torch.isfinite(scores).all()


def test_opq_mode_shares_code_parameters_across_items():
    ar = TinyAR()
    catalog = torch.tensor([[0, 0], [0, 1], [1, 0]])
    model = ExactMIPSDrafter(ar, catalog, 'opq')
    assert model.item_embeddings is None
    assert model.code_embeddings.shape == (2, 4, 6)
    assert model.catalog_item_vectors().shape == (3, 6)
