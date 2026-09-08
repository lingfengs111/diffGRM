"""Parallel coordinate/history attention and mixtures of structured item scores.

The legacy pooled encoder is unchanged by default. New heads share one history
encoding and one pairwise selector. Mixture aggregation happens AFTER complete
tuple scoring, so coordinates from different interests cannot form spurious
hybrid items. Auxiliary token predictions marginalize the interest axis.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .encoder_head_drafter import EncoderOnlyFourHeadDrafter
from .model import ModelOutput, make_norm
from .parallel_drafter import catalog_unary_scores


class HistoryAttentionDrafter(EncoderOnlyFourHeadDrafter):
    uses_full_history = True

    def __init__(self, config, dataset, tokenizer):
        super().__init__(config, dataset, tokenizer)
        self.history_head = str(config.get('history_head', 'attention'))
        self.n_interests = int(config.get('n_interests', 1))
        self.interest_temperature = float(config.get('interest_temperature', 1.0))
        if self.history_head not in ('attention', 'mlp'):
            raise ValueError('history_head must be attention|mlp')
        if self.n_interests < 1 or self.interest_temperature <= 0:
            raise ValueError('interest count and temperature must be positive')
        if self.history_head == 'mlp' and self.n_interests != 1:
            raise ValueError('MLP capacity control uses one interest')
        if self.history_head == 'attention':
            self.history_reader = nn.MultiheadAttention(
                self.n_embd, self.n_head,
                dropout=float(config['attn_pdrop']), batch_first=True,
            )
        else:
            # 4H^2 + 3H versus attention's 4H^2 + 4H parameters.
            self.history_reader = nn.Sequential(
                nn.Linear(self.n_embd, 2 * self.n_embd), nn.GELU(),
                nn.Linear(2 * self.n_embd, self.n_embd),
            )
        self.reader_norm = make_norm(
            config.get('norm_type', 'layernorm'), self.n_embd,
            float(config.get('norm_eps', 1e-5)),
        )
        self.history_reader.apply(self._init_weights)
        if self.n_interests > 1:
            self.interest_queries = nn.Parameter(
                torch.empty(self.n_interests, self.n_digit, self.n_embd)
            )
            nn.init.normal_(self.interest_queries, std=0.02)
            self.interest_gate = nn.Linear(self.n_embd, self.n_interests)
            nn.init.zeros_(self.interest_gate.weight)
            nn.init.zeros_(self.interest_gate.bias)
        self.last_interest_diagnostics = {}

    def forward_decoder_only(self, batch, **kwargs):
        del kwargs
        history = batch['encoder_hidden']
        valid = batch['history_mask'].bool()
        if history.ndim != 3 or valid.shape != history.shape[:2]:
            raise ValueError('expected history [B,S,H] and mask [B,S]')
        b = history.shape[0]
        last = valid.long().sum(1).sub(1).clamp_min(0)
        pooled = history[torch.arange(b, device=history.device), last]
        # Keep completely empty histories finite without exposing padding.
        pooled = pooled.masked_fill(~valid.any(1)[:, None], 0)
        query = torch.stack([p(pooled) for p in self.coordinate_projections], 1)
        query = query[:, None] + self.coordinate_queries[None, None]
        if self.n_interests > 1:
            query = query + self.interest_queries[None]
        query = F.gelu(query).reshape(b, self.n_interests * self.n_digit, self.n_embd)
        if self.history_head == 'attention':
            safe_valid = valid.clone()
            safe_valid[~safe_valid.any(1), 0] = True
            memory = history.masked_fill(~valid[..., None], 0)
            read, _ = self.history_reader(
                query, memory, memory, key_padding_mask=~safe_valid,
                need_weights=False,
            )
        else:
            read = self.history_reader(query)
        hidden = self.reader_norm(query + read)
        hidden = self.coordinate_norm(hidden + self.coordinate_mixer(hidden))
        hidden = hidden.reshape(b, self.n_interests, self.n_digit, self.n_embd)
        logits = torch.stack([
            self._compute_digit_logits(hidden[:, :, digit], digit)
            for digit in range(self.n_digit)
        ], dim=2)
        out = ModelOutput()
        if self.n_interests == 1:
            out.hidden_states, out.logits = hidden[:, 0], logits[:, 0]
        else:
            out.hidden_states, out.logits = hidden, logits
            out.interest_log_weights = F.log_softmax(self.interest_gate(pooled), -1)
        return out


def aggregate_interest_scores(scores, log_weights, temperature=1.0):
    """Weighted smooth max of complete energies; no item-wise normalization.

    scores: [B,M,N], normalized log_weights: [B,M]. This is an energy
    mixture, not a mixture of normalized full-catalog probabilities.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return temperature * torch.logsumexp(
        scores / temperature + log_weights[:, :, None], dim=1,
    )


def score_interest_catalog(model, decoded, catalog, selector=None):
    logits, hidden = decoded.logits, decoded.hidden_states
    b, m, d, v = logits.shape
    unary = catalog_unary_scores(logits.reshape(b * m, d, v), catalog)
    branch_scores = unary
    if selector is not None:
        branch_scores = branch_scores + selector(
            hidden.reshape(b * m, d, hidden.shape[-1]), catalog,
        )
    unary = unary.reshape(b, m, -1)
    branch_scores = branch_scores.reshape(b, m, -1)
    weights = decoded.interest_log_weights
    scores = aggregate_interest_scores(branch_scores, weights, model.interest_temperature)
    unary_scores = aggregate_interest_scores(unary, weights, model.interest_temperature)
    token_log_probs = torch.logsumexp(
        logits.log_softmax(-1) + weights[:, :, None, None], dim=1,
    )
    if not model.training:
        with torch.no_grad():
            keep = min(10, catalog.shape[0])
            top = scores.topk(keep, dim=-1).indices
            contributions = branch_scores / model.interest_temperature + weights[..., None]
            winners = contributions.gather(2, top[:, None].expand(-1, m, -1)).argmax(1)
            branch_top = branch_scores.topk(keep, dim=-1).indices
            overlaps = []
            for left in range(m):
                for right in range(left + 1, m):
                    intersection = (branch_top[:, left, :, None] == branch_top[:, right, None, :]).any(-1).sum(-1)
                    overlaps.append(intersection.float() / (2 * keep - intersection))
            model.last_interest_diagnostics = {
                'interest_gate_entropy': -(weights.exp() * weights).sum(1),
                'interest_branch_top10_jaccard': torch.stack(overlaps).mean(0),
                **{f'interest_{i}_top10_win_rate': winners.eq(i).float().mean(1) for i in range(m)},
                **{f'interest_{i}_gate_weight': weights[:, i].exp() for i in range(m)},
            }
    # Mixture scores are not additively factorized. The final field is only
    # the effective correction relative to the unary-only mixture.
    return scores, token_log_probs, hidden, unary_scores, scores - unary_scores
