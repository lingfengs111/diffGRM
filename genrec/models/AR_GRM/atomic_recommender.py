"""A fully atomic two-stage sequential recommender with no semantic IDs."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from genrec.models.AR_GRM.model import EncoderBlock


class AtomicSequentialRetriever(nn.Module):
    """Encode item-ID histories and score every catalog item directly."""

    def __init__(
        self,
        n_items: int,
        max_history_len: int,
        hidden_dim: int = 256,
        n_layer: int = 2,
        n_head: int = 4,
        n_inner: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_items = int(n_items)
        self.max_history_len = int(max_history_len)
        self.hidden_dim = int(hidden_dim)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_history_len, self.hidden_dim
        )
        self.dropout = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            EncoderBlock(
                self.hidden_dim,
                int(n_head),
                int(n_inner),
                float(dropout),
                float(dropout),
            )
            for _ in range(int(n_layer))
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.query_projection = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, history_ids: torch.Tensor) -> torch.Tensor:
        if history_ids.ndim != 2:
            raise ValueError('history_ids must have shape [B,S]')
        valid = history_ids.ne(0)
        if not valid.any(dim=1).all():
            raise ValueError('every history must contain at least one item')
        batch_size, sequence_length = history_ids.shape
        positions = torch.arange(sequence_length, device=history_ids.device)
        hidden = self.item_embedding(history_ids) + self.position_embedding(
            positions
        ).unsqueeze(0)
        hidden = self.dropout(hidden)

        causal = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=history_ids.device,
            )
        )
        attention_mask = causal.view(1, 1, sequence_length, sequence_length)
        attention_mask = attention_mask & valid[:, None, None, :]
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        hidden = self.final_norm(hidden)

        indices = torch.arange(sequence_length, device=history_ids.device)[None]
        last_positions = indices.masked_fill(~valid, -1).max(dim=1).values
        query = hidden[
            torch.arange(batch_size, device=history_ids.device), last_positions
        ]
        return self.query_projection(query)

    def catalog_scores(self, query: torch.Tensor) -> torch.Tensor:
        # Row zero is padding and is never a recommendable catalog item.
        item_vectors = self.item_embedding.weight[1:]
        return (
            self.logit_scale.exp().clamp(max=100.0)
            * F.normalize(query, dim=-1)
            @ F.normalize(item_vectors, dim=-1).transpose(0, 1)
        )

    def forward(self, history_ids: torch.Tensor):
        query = self.encode(history_ids)
        return self.catalog_scores(query), query


class AtomicCandidateRanker(nn.Module):
    """Proposal-aware non-generative verifier over atomic candidates."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.scorer = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        candidate_vectors: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 2 or candidate_vectors.ndim != 3:
            raise ValueError('expected query [B,H], candidates [B,K,H]')
        expanded = query[:, None, :].expand_as(candidate_vectors)
        features = torch.cat(
            [
                expanded,
                candidate_vectors,
                expanded * candidate_vectors,
                torch.abs(expanded - candidate_vectors),
            ],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)
