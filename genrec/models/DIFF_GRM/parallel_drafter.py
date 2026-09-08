"""One-pass catalog drafters for product-quantized semantic IDs.

The iterative DiffGRM decoder predicts the same four OPQ coordinates several
times while progressively revealing a path.  A proposal model does not need
to materialize that path: one full-mask decoder pass already returns a
distribution for every coordinate.  This module turns those distributions
into scores over the *legal item catalog* and optionally adds a lightweight
context-conditioned pairwise selector inspired by DFlash2.
"""

from itertools import combinations, permutations
import math

import torch
from torch import nn
import torch.nn.functional as F


def linear_curriculum_weight(
    epoch: int,
    start: float,
    end: float,
    decay_epochs: int,
):
    """Linearly anneal an objective weight over one-indexed epochs.

    The first training epoch uses ``start`` and ``decay_epochs`` uses ``end``;
    later epochs remain at ``end``.  Keeping this helper independent of the
    training script makes the exact Domino-style schedule easy to test and
    record in experiment protocols.
    """
    if epoch < 1:
        raise ValueError("epoch must be one-indexed and positive")
    if decay_epochs < 1:
        raise ValueError("decay_epochs must be positive")
    if decay_epochs == 1:
        return float(end)
    progress = min(max((epoch - 1) / float(decay_epochs - 1), 0.0), 1.0)
    return float(start) + progress * (float(end) - float(start))


def catalog_unary_scores(logits: torch.Tensor, catalog_codes: torch.Tensor):
    """Score every legal catalog row from parallel coordinate logits.

    Args:
        logits: ``[batch, n_digit, codebook_size]``.
        catalog_codes: ``[n_item, n_digit]`` raw codebook indices.

    Returns:
        Log-probability sums with shape ``[batch, n_item]``.
    """
    if logits.ndim != 3 or catalog_codes.ndim != 2:
        raise ValueError("expected logits [B,D,K] and catalog_codes [N,D]")
    if logits.shape[1] != catalog_codes.shape[1]:
        raise ValueError("logits and catalog codes use different digit counts")
    if catalog_codes.numel() and (
        catalog_codes.min() < 0 or catalog_codes.max() >= logits.shape[-1]
    ):
        raise ValueError("catalog code lies outside the decoder vocabulary")

    log_probs = F.log_softmax(logits, dim=-1)
    scores = logits.new_zeros(logits.shape[0], catalog_codes.shape[0])
    for digit in range(logits.shape[1]):
        scores = scores + log_probs[:, digit, catalog_codes[:, digit]]
    return scores


def batched_catalog_unary_scores(
    logits: torch.Tensor,
    candidate_codes: torch.Tensor,
):
    """Score a different sampled legal-item set for every example.

    ``logits`` has shape ``[B,D,K]`` and ``candidate_codes`` has shape
    ``[B,S,D]``.  Unlike :func:`catalog_unary_scores`, this avoids expanding a
    shared full catalog when training with per-example sampled negatives.
    """
    if logits.ndim != 3 or candidate_codes.ndim != 3:
        raise ValueError(
            "expected logits [B,D,K] and candidate_codes [B,S,D]"
        )
    if logits.shape[0] != candidate_codes.shape[0]:
        raise ValueError("logits and candidate codes use different batches")
    if logits.shape[1] != candidate_codes.shape[2]:
        raise ValueError("logits and candidate codes use different digit counts")
    if candidate_codes.numel() and (
        candidate_codes.min() < 0
        or candidate_codes.max() >= logits.shape[-1]
    ):
        raise ValueError("candidate code lies outside the decoder vocabulary")

    log_probs = F.log_softmax(logits, dim=-1)
    scores = logits.new_zeros(logits.shape[0], candidate_codes.shape[1])
    for digit in range(logits.shape[1]):
        scores = scores + log_probs[:, digit].gather(
            1, candidate_codes[:, :, digit]
        )
    return scores


def code_rows(
    codes: torch.Tensor,
    catalog_codes: torch.Tensor,
    codebook_size: int,
    identity_start_digit: int = 0,
):
    """Map collision-free item-identity tuples to catalog rows on device.

    A tokenizer may prepend a nuisance/path token, such as Latte's randomly
    sampled latent token, before the digits that identify an item.  Training
    targets can use any legal latent value while the catalog stores a single
    canonical path, so those prefix digits must be excluded from lookup.
    """
    if codes.ndim != 2 or catalog_codes.ndim != 2:
        raise ValueError("expected code tensors with rank two")
    if codes.shape[1] != catalog_codes.shape[1]:
        raise ValueError("target and catalog digit counts differ")
    identity_start_digit = int(identity_start_digit)
    if not 0 <= identity_start_digit < codes.shape[1]:
        raise ValueError("identity_start_digit lies outside the code tuple")
    codes = codes[:, identity_start_digit:]
    catalog_codes = catalog_codes[:, identity_start_digit:]
    multipliers = torch.tensor(
        [codebook_size ** power for power in reversed(range(codes.shape[1]))],
        dtype=torch.long,
        device=codes.device,
    )
    target_keys = (codes.long() * multipliers).sum(dim=1)
    catalog_keys = (catalog_codes.long() * multipliers).sum(dim=1)
    sorted_keys, sorted_rows = torch.sort(catalog_keys)
    positions = torch.searchsorted(sorted_keys, target_keys)
    if positions.numel() and (
        positions.max() >= sorted_keys.numel()
        or not torch.equal(sorted_keys[positions], target_keys)
    ):
        raise ValueError("at least one target SID is absent from the catalog")
    return sorted_rows[positions]


def candidate_tree_order_costs(candidates: torch.Tensor, codebook_size: int):
    """Count unique trie nodes for every coordinate permutation.

    The complete candidate SIDs are already known, so choosing a verification
    order does not alter recall.  A lower node count means more prefixes can be
    shared by a future tree-attention/cached AR verifier.
    """
    if candidates.ndim != 3:
        raise ValueError("candidates must have shape [B,C,D]")
    orders = tuple(permutations(range(candidates.shape[-1])))
    costs = []
    values = candidates.long()
    for order in orders:
        prefix = torch.zeros_like(values[:, :, 0])
        node_count = torch.zeros(
            values.shape[0], dtype=torch.long, device=values.device
        )
        for digit in order:
            prefix = prefix * int(codebook_size) + values[:, :, digit]
            sorted_prefix = prefix.sort(dim=1).values
            unique = torch.ones_like(sorted_prefix, dtype=torch.bool)
            unique[:, 1:] = sorted_prefix[:, 1:] != sorted_prefix[:, :-1]
            node_count = node_count + unique.sum(dim=1)
        costs.append(node_count)
    return torch.stack(costs, dim=1), orders


def candidate_tree_diagnostics(candidates: torch.Tensor, codebook_size: int):
    """Return per-query potential computation savings from adaptive order."""
    costs, orders = candidate_tree_order_costs(candidates, codebook_size)
    fixed_index = orders.index(tuple(range(candidates.shape[-1])))
    fixed = costs[:, fixed_index]
    best, best_order = costs.min(dim=1)
    independent = torch.full_like(
        best, candidates.shape[1] * candidates.shape[2]
    )
    return {
        "independent_nodes": independent,
        "fixed_order_nodes": fixed,
        "best_order_nodes": best,
        "best_vs_fixed_ratio": best.float() / fixed.clamp_min(1).float(),
        "best_vs_independent_ratio": best.float() / independent.clamp_min(1).float(),
        "best_order_index": best_order,
        "orders": orders,
    }


class PairwisePathSelector(nn.Module):
    """Parallel low-rank compatibility over OPQ coordinate pairs/triples.

    DFlash2 scores adjacent token candidates after its one-pass draft.  OPQ
    coordinates have no natural adjacency, so this selector uses the complete
    graph over coordinate pairs.  An optional CP-factorized triple residual
    measures dependencies that no sum of pair energies can represent.  All
    scores use the same decoder pass; neither variant adds an iterative reveal
    or another backbone call.
    """

    def __init__(
        self,
        n_digit: int,
        codebook_size: int,
        context_dim: int,
        rank: int = 32,
        triple_rank: int = 0,
    ):
        super().__init__()
        self.n_digit = int(n_digit)
        self.codebook_size = int(codebook_size)
        self.rank = int(rank)
        self.triple_rank = int(triple_rank)
        if self.rank <= 0 or self.triple_rank < 0:
            raise ValueError("rank must be positive and triple_rank non-negative")
        self.pairs = tuple(combinations(range(self.n_digit), 2))
        self.triples = tuple(combinations(range(self.n_digit), 3))
        self.left = nn.Parameter(
            torch.empty(self.n_digit, self.codebook_size, self.rank)
        )
        self.right = nn.Parameter(
            torch.empty(self.n_digit, self.codebook_size, self.rank)
        )
        self.gates = nn.ModuleList(
            nn.Linear(2 * int(context_dim), self.rank) for _ in self.pairs
        )
        self.pair_scale = nn.Parameter(torch.tensor(0.05))
        nn.init.normal_(self.left, std=0.02)
        nn.init.normal_(self.right, std=0.02)
        for gate in self.gates:
            nn.init.zeros_(gate.bias)

        if self.triple_rank:
            # Three role-specific factors implement a low-rank CP tensor for
            # each sorted coordinate triple without materializing K^3 tables.
            self.triple_factors = nn.Parameter(
                torch.empty(
                    3,
                    self.n_digit,
                    self.codebook_size,
                    self.triple_rank,
                )
            )
            self.triple_gates = nn.ModuleList(
                nn.Linear(3 * int(context_dim), self.triple_rank)
                for _ in self.triples
            )
            # Zero initialization makes a pairwise checkpoint an exact
            # functional warm start.  The scale receives a gradient on the
            # first update; factors and gates then begin adapting.
            self.triple_scale = nn.Parameter(torch.tensor(0.0))
            nn.init.normal_(self.triple_factors, std=0.02)
            for gate in self.triple_gates:
                nn.init.zeros_(gate.bias)
        else:
            self.register_parameter('triple_factors', None)
            self.triple_gates = nn.ModuleList()
            self.register_parameter('triple_scale', None)

    def forward(self, decoder_hidden: torch.Tensor, catalog_codes: torch.Tensor):
        if decoder_hidden.ndim != 3:
            raise ValueError("decoder_hidden must have shape [B,D,H]")
        if decoder_hidden.shape[1] != self.n_digit:
            raise ValueError("decoder hidden digit count does not match selector")
        if catalog_codes.ndim not in (2, 3):
            raise ValueError(
                "catalog codes must have shape [N,D] or sampled [B,S,D]"
            )
        batched_candidates = catalog_codes.ndim == 3
        if batched_candidates:
            if catalog_codes.shape[0] != decoder_hidden.shape[0]:
                raise ValueError(
                    "sampled catalog and decoder hidden use different batches"
                )
            if catalog_codes.shape[2] != self.n_digit:
                raise ValueError("sampled catalog digit count does not match")
            n_candidates = catalog_codes.shape[1]
        else:
            if catalog_codes.shape[1] != self.n_digit:
                raise ValueError("catalog digit count does not match")
            n_candidates = catalog_codes.shape[0]
        pair_scores = decoder_hidden.new_zeros(
            decoder_hidden.shape[0], n_candidates
        )
        scale = math.sqrt(self.rank)
        for pair_idx, (left_digit, right_digit) in enumerate(self.pairs):
            context = torch.cat(
                [
                    decoder_hidden[:, left_digit],
                    decoder_hidden[:, right_digit],
                ],
                dim=-1,
            )
            gate = torch.tanh(self.gates[pair_idx](context))
            if batched_candidates:
                left = self.left[
                    left_digit, catalog_codes[:, :, left_digit]
                ]
                right = self.right[
                    right_digit, catalog_codes[:, :, right_digit]
                ]
            else:
                left = self.left[left_digit, catalog_codes[:, left_digit]]
                right = self.right[right_digit, catalog_codes[:, right_digit]]
            catalog_features = left * right
            if batched_candidates:
                contribution = torch.einsum(
                    "br,bsr->bs", gate, catalog_features
                )
            else:
                contribution = torch.einsum(
                    "br,nr->bn", gate, catalog_features
                )
            pair_scores = pair_scores + contribution / scale
        scores = self.pair_scale * pair_scores
        if self.triple_rank:
            triple_scores = decoder_hidden.new_zeros(
                decoder_hidden.shape[0], n_candidates
            )
            scale = math.sqrt(self.triple_rank)
            for triple_idx, digits in enumerate(self.triples):
                context = torch.cat(
                    [decoder_hidden[:, digit] for digit in digits], dim=-1
                )
                gate = torch.tanh(self.triple_gates[triple_idx](context))
                feature_shape = (
                    (decoder_hidden.shape[0], n_candidates, self.triple_rank)
                    if batched_candidates
                    else (n_candidates, self.triple_rank)
                )
                catalog_features = torch.ones(
                    *feature_shape,
                    device=decoder_hidden.device,
                    dtype=decoder_hidden.dtype,
                )
                for role, digit in enumerate(digits):
                    indices = (
                        catalog_codes[:, :, digit]
                        if batched_candidates else catalog_codes[:, digit]
                    )
                    catalog_features = catalog_features * self.triple_factors[
                        role, digit, indices
                    ]
                if batched_candidates:
                    contribution = torch.einsum(
                        "br,bsr->bs", gate, catalog_features
                    )
                else:
                    contribution = torch.einsum(
                        "br,nr->bn", gate, catalog_features
                    )
                triple_scores = triple_scores + contribution / scale
            scores = scores + self.triple_scale * triple_scores
        return scores


class CausalResidualCorrector(nn.Module):
    """Domino-style causal residual over a parallel SID prediction.

    The expensive history-conditioned backbone still runs exactly once.  For
    each complete legal candidate, this small head teacher-forces its prefix
    through a GRU and adds a low-rank correction to the parallel logits.  The
    same correction can therefore be trained on ground-truth paths and used to
    rerank a moderately wide candidate pool without another backbone pass.

    Coordinate-specific embeddings are intentional: OPQ coordinate values
    live in different codebooks, so e.g. value 7 at levels 0 and 1 is not the
    same token.  The final projection is zero-initialized, making construction
    an exact functional copy of the base token distribution.
    """

    def __init__(
        self,
        n_digit: int,
        codebook_size: int,
        context_dim: int,
        state_dim: int = 64,
        rank: int = 32,
    ):
        super().__init__()
        self.n_digit = int(n_digit)
        self.codebook_size = int(codebook_size)
        self.context_dim = int(context_dim)
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        if min(
            self.n_digit,
            self.codebook_size,
            self.context_dim,
            self.state_dim,
            self.rank,
        ) <= 0:
            raise ValueError('all corrector dimensions must be positive')

        self.root_embedding = nn.Parameter(torch.empty(self.state_dim))
        self.code_embeddings = nn.ModuleList(
            nn.Embedding(self.codebook_size, self.state_dim)
            for _ in range(self.n_digit)
        )
        self.prefix_gru = nn.GRUCell(self.state_dim, self.state_dim)
        self.correction_in = nn.ModuleList(
            nn.Linear(self.context_dim + self.state_dim, self.rank)
            for _ in range(self.n_digit)
        )
        self.correction_out = nn.ModuleList(
            nn.Linear(self.rank, self.codebook_size, bias=False)
            for _ in range(self.n_digit)
        )

        nn.init.normal_(self.root_embedding, std=0.02)
        for embedding in self.code_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        for projection in self.correction_out:
            nn.init.zeros_(projection.weight)

    def forward(
        self,
        base_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        candidate_codes: torch.Tensor,
        return_logits: bool = True,
    ):
        """Score teacher-forced candidate paths under corrected logits.

        Args:
            base_hidden: parallel decoder states ``[B,D,H]``.
            base_logits: parallel coordinate logits ``[B,D,K]``.
            candidate_codes: raw legal codes ``[B,C,D]`` or targets ``[B,D]``.

        Returns:
            A dictionary containing path/token log-probabilities and residual
            logits.  Rank-two targets retain a singleton candidate dimension;
            this keeps training and candidate reranking on one code path.
        """
        if base_hidden.ndim != 3 or base_logits.ndim != 3:
            raise ValueError('base tensors must have shape [B,D,H/K]')
        if candidate_codes.ndim == 2:
            candidate_codes = candidate_codes[:, None, :]
        if candidate_codes.ndim != 3:
            raise ValueError('candidate codes must have shape [B,C,D] or [B,D]')
        batch_size, n_candidates, n_digit = candidate_codes.shape
        if base_hidden.shape[:2] != (batch_size, self.n_digit):
            raise ValueError('base hidden shape does not match the corrector')
        if base_hidden.shape[-1] != self.context_dim:
            raise ValueError('base hidden width does not match context_dim')
        if base_logits.shape != (
            batch_size, self.n_digit, self.codebook_size
        ):
            raise ValueError('base logits shape does not match the corrector')
        if n_digit != self.n_digit:
            raise ValueError('candidate digit count does not match the corrector')
        if candidate_codes.numel() and (
            candidate_codes.min() < 0
            or candidate_codes.max() >= self.codebook_size
        ):
            raise ValueError('candidate code lies outside the codebook')

        # A learned root plays the role of Domino's last verified token.  The
        # history-dependent information itself stays in the parallel states.
        flat_size = batch_size * n_candidates
        state = self.root_embedding.unsqueeze(0).expand(flat_size, -1)
        token_log_probs = []
        residual_logits = [] if return_logits else None
        corrected_logits = [] if return_logits else None
        for digit in range(self.n_digit):
            context = base_hidden[:, None, digit, :].expand(
                -1, n_candidates, -1
            ).reshape(flat_size, self.context_dim)
            residual = self.correction_out[digit](
                F.silu(self.correction_in[digit](torch.cat([context, state], dim=-1)))
            )
            logits = base_logits[:, None, digit, :].expand(
                -1, n_candidates, -1
            ).reshape(flat_size, self.codebook_size) + residual
            flat_codes = candidate_codes[:, :, digit].reshape(flat_size)
            token_log_probs.append(
                F.log_softmax(logits.float(), dim=-1).gather(
                    1, flat_codes[:, None]
                ).squeeze(1).reshape(batch_size, n_candidates)
            )
            if return_logits:
                residual_logits.append(
                    residual.reshape(
                        batch_size, n_candidates, self.codebook_size
                    )
                )
                corrected_logits.append(
                    logits.reshape(
                        batch_size, n_candidates, self.codebook_size
                    )
                )
            if digit + 1 < self.n_digit:
                state = self.prefix_gru(
                    self.code_embeddings[digit](flat_codes), state
                )

        token_log_probs = torch.stack(token_log_probs, dim=-1)
        output = {
            'path_log_probs': token_log_probs.sum(dim=-1),
            'token_log_probs': token_log_probs,
        }
        if return_logits:
            output['residual_logits'] = torch.stack(residual_logits, dim=2)
            output['corrected_logits'] = torch.stack(corrected_logits, dim=2)
        return output
