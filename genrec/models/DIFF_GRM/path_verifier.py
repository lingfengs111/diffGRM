"""Lightweight parallel verification of complete semantic-ID candidates."""

from __future__ import annotations

import torch
from torch import nn


def masked_history_mean(hidden: torch.Tensor, history_mask: torch.Tensor | None):
    if history_mask is None:
        return hidden.mean(dim=1)
    mask = history_mask.to(hidden.device).bool()
    if mask.ndim != 2 or mask.shape != hidden.shape[:2]:
        raise ValueError('history_mask must have shape [B,S]')
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def last_valid_history(hidden: torch.Tensor, history_mask: torch.Tensor | None):
    """Return the final observed item state without averaging away recency."""
    if history_mask is None:
        return hidden[:, -1]
    mask = history_mask.to(hidden.device).bool()
    if mask.ndim != 2 or mask.shape != hidden.shape[:2]:
        raise ValueError('history_mask must have shape [B,S]')
    last = mask.long().sum(dim=1).sub(1).clamp_min(0)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class IndependentSIDHistoryEncoder(nn.Module):
    """A separate causal SID history backbone for parameter-matched controls.

    The lightweight verifier normally reuses the frozen drafter states.  This
    module deliberately pays for a second history pass, matching the setting
    of the standalone AR verifier without introducing autoregressive path
    factorization.  Each OPQ coordinate has its own embedding table so equal
    raw code values at different coordinates remain distinct.
    """

    def __init__(
        self,
        n_digit: int,
        codebook_size: int,
        hidden_dim: int,
        n_head: int,
        layers: int,
        max_history_len: int,
        dropout: float = 0.1,
        inner_dim: int | None = None,
    ):
        super().__init__()
        self.n_digit = int(n_digit)
        self.codebook_size = int(codebook_size)
        self.hidden_dim = int(hidden_dim)
        self.max_history_len = int(max_history_len)
        inner_dim = int(inner_dim or 2 * self.hidden_dim)
        if self.hidden_dim % int(n_head):
            raise ValueError('history hidden_dim must be divisible by n_head')
        if int(layers) < 1:
            raise ValueError('independent history encoder needs at least one layer')

        self.code_embeddings = nn.Parameter(
            torch.empty(
                self.n_digit,
                self.codebook_size,
                self.hidden_dim,
            )
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.positions = nn.Embedding(self.max_history_len, self.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(n_head),
            dim_feedforward=inner_dim,
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.code_embeddings, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)

    def forward(self, history_sid: torch.Tensor):
        if history_sid.ndim != 3 or history_sid.shape[-1] != self.n_digit:
            raise ValueError('history_sid must have shape [B,S,D]')
        if history_sid.shape[1] > self.max_history_len:
            raise ValueError('history exceeds max_history_len')
        valid = history_sid.ne(-1).any(dim=-1)
        clean = history_sid.clamp(min=0, max=self.codebook_size - 1).long()
        coordinate_embeddings = torch.stack(
            [
                self.code_embeddings[digit, clean[:, :, digit]]
                for digit in range(self.n_digit)
            ],
            dim=2,
        )
        coordinate_embeddings = coordinate_embeddings * valid[
            :, :, None, None
        ].to(coordinate_embeddings.dtype)
        hidden = self.item_mlp(
            coordinate_embeddings.flatten(start_dim=2)
        )
        positions = torch.arange(history_sid.shape[1], device=history_sid.device)
        hidden = self.dropout(hidden + self.positions(positions)[None])
        causal_mask = torch.triu(
            torch.ones(
                history_sid.shape[1],
                history_sid.shape[1],
                dtype=torch.bool,
                device=history_sid.device,
            ),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~valid,
        )
        hidden = self.output_norm(hidden)
        return hidden * valid.unsqueeze(-1).to(hidden.dtype)


class MLPPathMixer(nn.Module):
    """Parameter-matched non-attention control over the complete code tuple.

    Flattening four coordinate embeddings already preserves coordinate identity.
    The input projection contributes roughly ``D*H^2`` parameters and each
    residual FFN contributes ``8*H^2``.  With four digits and one block this is
    close to the ``12*H^2`` weight budget of one Transformer encoder layer.
    """

    def __init__(self, n_digit: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.n_digit = int(n_digit)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Linear(
            self.n_digit * self.hidden_dim, self.hidden_dim
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                nn.Dropout(float(dropout)),
            )
            for _ in range(int(layers))
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, coordinate_tokens: torch.Tensor):
        hidden = self.input_projection(coordinate_tokens.flatten(start_dim=1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output_norm(hidden)


class ParallelPathVerifier(nn.Module):
    """Score all candidate SID paths without autoregressive factorization.

    A short bidirectional coordinate mixer models the four codes within each
    path.  A permutation-equivariant set mixer then lets competing candidates
    calibrate one another.  The model reuses the drafter history encoding and
    therefore adds no second history backbone pass.
    """

    def __init__(
        self,
        n_digit: int,
        codebook_size: int,
        context_dim: int,
        hidden_dim: int = 128,
        n_head: int = 4,
        coordinate_layers: int = 1,
        set_layers: int = 1,
        dropout: float = 0.1,
        coordinate_mode: str = 'bidirectional',
        history_pooling: str = 'mean',
    ):
        super().__init__()
        self.n_digit = int(n_digit)
        self.codebook_size = int(codebook_size)
        self.hidden_dim = int(hidden_dim)
        self.coordinate_mode = str(coordinate_mode)
        self.history_pooling = str(history_pooling)
        if self.hidden_dim % int(n_head):
            raise ValueError('hidden_dim must be divisible by n_head')
        if self.coordinate_mode not in ('bidirectional', 'causal', 'mlp'):
            raise ValueError(
                'coordinate_mode must be bidirectional, causal, or mlp'
            )
        if self.history_pooling not in ('mean', 'last'):
            raise ValueError('history_pooling must be mean or last')

        self.code_embeddings = nn.Parameter(
            torch.empty(
                self.n_digit,
                self.codebook_size,
                self.hidden_dim,
            )
        )
        self.coordinate_positions = nn.Parameter(
            torch.empty(self.n_digit, self.hidden_dim)
        )
        nn.init.normal_(self.code_embeddings, std=0.02)
        nn.init.normal_(self.coordinate_positions, std=0.02)

        if self.coordinate_mode == 'mlp':
            self.coordinate_mixer = MLPPathMixer(
                self.n_digit,
                self.hidden_dim,
                coordinate_layers,
                dropout,
            )
        else:
            coordinate_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(n_head),
                dim_feedforward=4 * self.hidden_dim,
                dropout=float(dropout),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.coordinate_mixer = nn.TransformerEncoder(
                coordinate_layer, num_layers=int(coordinate_layers)
            )
        self.context_projection = nn.Sequential(
            nn.Linear(int(context_dim), self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.proposal_projection = nn.Linear(1, self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        if int(set_layers):
            set_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(n_head),
                dim_feedforward=4 * self.hidden_dim,
                dropout=float(dropout),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.set_mixer = nn.TransformerEncoder(
                set_layer, num_layers=int(set_layers)
            )
        else:
            self.set_mixer = nn.Identity()
        self.score = nn.Linear(self.hidden_dim, 1)

    @staticmethod
    def _standardize(scores: torch.Tensor):
        return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)

    def forward(
        self,
        history_hidden: torch.Tensor,
        candidates: torch.Tensor,
        proposal_scores: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ):
        if candidates.ndim != 3:
            raise ValueError('candidates must have shape [B,C,D]')
        if candidates.shape[-1] != self.n_digit:
            raise ValueError('candidate digit count does not match verifier')
        if proposal_scores.shape != candidates.shape[:2]:
            raise ValueError('proposal_scores must have shape [B,C]')
        if candidates.numel() and (
            candidates.min() < 0 or candidates.max() >= self.codebook_size
        ):
            raise ValueError('candidate code lies outside the codebook')

        batch_size, n_candidates, _ = candidates.shape
        coordinate_tokens = torch.stack(
            [
                self.code_embeddings[digit, candidates[:, :, digit]]
                for digit in range(self.n_digit)
            ],
            dim=2,
        )
        coordinate_tokens = coordinate_tokens + self.coordinate_positions[
            None, None
        ]
        flat_coordinate_tokens = coordinate_tokens.reshape(
            batch_size * n_candidates,
            self.n_digit,
            self.hidden_dim,
        )
        if self.coordinate_mode == 'mlp':
            path_hidden = self.coordinate_mixer(flat_coordinate_tokens)
        else:
            coordinate_mask = None
            if self.coordinate_mode == 'causal':
                coordinate_mask = torch.triu(
                    torch.ones(
                        self.n_digit,
                        self.n_digit,
                        dtype=torch.bool,
                        device=coordinate_tokens.device,
                    ),
                    diagonal=1,
                )
            path_hidden = self.coordinate_mixer(
                flat_coordinate_tokens,
                mask=coordinate_mask,
            ).mean(dim=1)
        path_hidden = path_hidden.reshape(
            batch_size, n_candidates, self.hidden_dim
        )

        if self.history_pooling == 'last':
            pooled_history = last_valid_history(history_hidden, history_mask)
        else:
            pooled_history = masked_history_mean(history_hidden, history_mask)
        context = self.context_projection(pooled_history)[:, None, :].expand(
            -1, n_candidates, -1
        )
        proposal = self.proposal_projection(
            self._standardize(proposal_scores).unsqueeze(-1)
        )
        fused = self.fusion(
            torch.cat(
                [path_hidden, context, path_hidden * context, proposal],
                dim=-1,
            )
        )
        fused = self.set_mixer(fused)
        return self.score(fused).squeeze(-1)
