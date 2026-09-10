"""Matched catalog drafters for atomic-item versus semantic-ID controls.

Every variant consumes the same frozen AR history states and uses the same
pooling/query trunk.  The only intended difference is the catalog
parameterization: an independent item table, four unary code tables, or the
same unary model plus low-rank pairwise code compatibility.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector


class MatchedCatalogDrafter(nn.Module):
    """Full-catalog scorer with a controlled representation choice."""

    def __init__(
        self,
        ar_model,
        catalog_codes: torch.Tensor,
        representation: str,
        pair_rank: int = 32,
    ):
        super().__init__()
        if representation not in {'atomic', 'unary', 'pairwise'}:
            raise ValueError('representation must be atomic|unary|pairwise')
        if catalog_codes.ndim != 2:
            raise ValueError('catalog_codes must have shape [n_item,n_digit]')
        if catalog_codes.shape[1] != ar_model.n_digit:
            raise ValueError('catalog digit count differs from AR model')

        self.representation = representation
        self.n_digit = int(ar_model.n_digit)
        self.codebook_size = int(ar_model.codebook_size)
        self.hidden_dim = int(ar_model.n_embd)
        self.sid_offset = int(ar_model.tokenizer.sid_offset)
        self.register_buffer('catalog_codes', catalog_codes.long().clone())

        # This trunk is deliberately constructed before every variant-specific
        # parameter so a fixed seed gives byte-identical common initialization.
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
            tables = []
            for digit in range(self.n_digit):
                token_ids = (
                    torch.arange(
                        self.codebook_size,
                        device=catalog_codes.device,
                    )
                    + self.sid_offset
                    + digit * self.codebook_size
                )
                tables.append(ar_model.embedding(token_ids).detach().clone())
            initial_code_tables = torch.stack(tables, dim=0)
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

        if representation == 'atomic':
            self.item_embeddings = nn.Parameter(initial_items.clone())
            self.code_embeddings = None
            self.digit_projections = None
            self.digit_offsets = None
            self.selector = None
        else:
            self.item_embeddings = None
            self.code_embeddings = nn.Parameter(initial_code_tables)
            self.digit_projections = nn.ModuleList(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for _ in range(self.n_digit)
            )
            self.digit_offsets = nn.Parameter(
                torch.empty(self.n_digit, self.hidden_dim)
            )
            nn.init.normal_(self.digit_offsets, std=0.02)
            self.selector = (
                PairwisePathSelector(
                    n_digit=self.n_digit,
                    codebook_size=self.codebook_size,
                    context_dim=self.hidden_dim,
                    rank=int(pair_rank),
                )
                if representation == 'pairwise'
                else None
            )

    def pool_history(
        self,
        encoder_hidden: torch.Tensor,
        history_sid: torch.Tensor,
    ) -> torch.Tensor:
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

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        history_sid: torch.Tensor,
    ) -> torch.Tensor:
        query = self.pool_history(encoder_hidden, history_sid)
        scale = self.logit_scale.exp().clamp(max=100.0)
        if self.representation == 'atomic':
            return scale * F.normalize(query, dim=-1) @ F.normalize(
                self.item_embeddings, dim=-1
            ).transpose(0, 1)

        digit_hidden = torch.stack(
            [
                projection(query) + self.digit_offsets[digit]
                for digit, projection in enumerate(self.digit_projections)
            ],
            dim=1,
        )
        logits = scale * torch.einsum(
            'bdh,dkh->bdk',
            F.normalize(digit_hidden, dim=-1),
            F.normalize(self.code_embeddings, dim=-1),
        )
        log_probs = F.log_softmax(logits, dim=-1)
        scores = logits.new_zeros(logits.shape[0], self.catalog_codes.shape[0])
        for digit in range(self.n_digit):
            scores = scores + log_probs[
                :, digit, self.catalog_codes[:, digit]
            ]
        if self.selector is not None:
            scores = scores + self.selector(digit_hidden, self.catalog_codes)
        return scores

    def parameter_report(self):
        groups = {
            'common_trunk': list(self.pool_norm.parameters())
            + list(self.pool_score.parameters())
            + list(self.query_projection.parameters())
            + [self.logit_scale],
            'atomic_table': [self.item_embeddings]
            if self.item_embeddings is not None else [],
            'code_tables': [self.code_embeddings]
            if self.code_embeddings is not None else [],
            'digit_heads': (
                list(self.digit_projections.parameters()) + [self.digit_offsets]
                if self.digit_projections is not None else []
            ),
            'pairwise_selector': list(self.selector.parameters())
            if self.selector is not None else [],
        }
        report = {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in groups.items()
        }
        report['trainable_total'] = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        return report
