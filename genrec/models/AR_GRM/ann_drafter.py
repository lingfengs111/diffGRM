"""Exact-MIPS drafters over item or compositional OPQ representations.

The current catalogs are small enough to score every item exactly.  This keeps
the retrieval model and the approximate index separate: after establishing the
quality ceiling, the same item vectors can be served by an ANN library.
"""

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


class ExactMIPSDrafter(nn.Module):
    """Map frozen history features to a query and retrieve catalog items.

    ``item_mode='id'`` gives every item an independent trainable vector.
    ``item_mode='opq'`` constructs item vectors from four shared code tables and
    the AR model's item-composition MLP, retaining OPQ's parameter sharing.
    Both variants start from the exact same AR item representation.
    """

    def __init__(self, ar_model, catalog_codes: torch.Tensor, item_mode: str):
        super().__init__()
        if item_mode not in {'id', 'opq'}:
            raise ValueError("item_mode must be 'id' or 'opq'")
        if catalog_codes.ndim != 2:
            raise ValueError('catalog_codes must have shape [n_item,n_digit]')
        if catalog_codes.shape[1] != ar_model.n_digit:
            raise ValueError('catalog digit count differs from AR model')

        self.item_mode = item_mode
        self.n_digit = int(ar_model.n_digit)
        self.codebook_size = int(ar_model.codebook_size)
        self.hidden_dim = int(ar_model.n_embd)
        self.sid_offset = int(ar_model.tokenizer.sid_offset)
        self.register_buffer('catalog_codes', catalog_codes.long().clone())

        self.pool_norm = nn.LayerNorm(self.hidden_dim)
        self.pool_score = nn.Linear(self.hidden_dim, 1, bias=False)
        self.query_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        with torch.no_grad():
            code_tables = []
            for digit in range(self.n_digit):
                token_ids = (
                    torch.arange(self.codebook_size, device=catalog_codes.device)
                    + self.sid_offset
                    + digit * self.codebook_size
                )
                code_tables.append(ar_model.embedding(token_ids).detach().clone())
            initial_code_tables = torch.stack(code_tables, dim=0)
            gathered = torch.stack(
                [
                    initial_code_tables[digit, catalog_codes[:, digit]]
                    for digit in range(self.n_digit)
                ],
                dim=1,
            )
            initial_items = ar_model.item_mlp(
                gathered.reshape(catalog_codes.shape[0], -1)
            ).detach()

        if self.item_mode == 'id':
            self.item_embeddings = nn.Parameter(initial_items.clone())
            self.code_embeddings = None
            self.item_projection = None
        else:
            self.item_embeddings = None
            self.code_embeddings = nn.Parameter(initial_code_tables)
            self.item_projection = copy.deepcopy(ar_model.item_mlp)

    def pool_history(self, encoder_hidden: torch.Tensor, history_sid: torch.Tensor):
        if encoder_hidden.ndim != 3 or history_sid.ndim != 3:
            raise ValueError('expected encoder_hidden/history_sid with rank three')
        valid = history_sid.ne(-1).any(dim=-1)
        if not valid.any(dim=1).all():
            raise ValueError('every history must contain at least one valid item')

        normalized = self.pool_norm(encoder_hidden)
        attention_logits = self.pool_score(normalized).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid, float('-inf'))
        attention = F.softmax(attention_logits, dim=-1)
        attended = torch.einsum('bs,bsd->bd', attention, encoder_hidden)

        positions = torch.arange(valid.shape[1], device=valid.device)[None]
        last_positions = positions.masked_fill(~valid, -1).max(dim=1).values
        last = encoder_hidden[
            torch.arange(encoder_hidden.shape[0], device=encoder_hidden.device),
            last_positions,
        ]
        return self.query_projection(torch.cat([last, attended], dim=-1))

    def catalog_item_vectors(self):
        if self.item_mode == 'id':
            return self.item_embeddings
        gathered = torch.stack(
            [
                self.code_embeddings[digit, self.catalog_codes[:, digit]]
                for digit in range(self.n_digit)
            ],
            dim=1,
        )
        return self.item_projection(gathered.reshape(gathered.shape[0], -1))

    def forward(self, encoder_hidden: torch.Tensor, history_sid: torch.Tensor):
        query = F.normalize(self.pool_history(encoder_hidden, history_sid), dim=-1)
        items = F.normalize(self.catalog_item_vectors(), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * query @ items.transpose(0, 1)

