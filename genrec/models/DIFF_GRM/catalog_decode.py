"""Catalog-constrained, order-marginalized decoding for DiffGRM.

Unlike a left-to-right trie, a state here is an arbitrary partial assignment
of SID digits.  Different reveal orders that reach the same partial assignment
are merged with log-sum-exp, matching the order-marginalized path objective.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F


class CatalogConstraint:
    """Index legal next (digit, code) transitions for partial SID states."""

    def __init__(self, catalog_codes, codebook_size: int):
        codes = torch.as_tensor(catalog_codes, dtype=torch.long).cpu()
        if codes.ndim != 2:
            raise ValueError("catalog_codes must have shape [num_items, n_digit]")
        self.n_digit = int(codes.shape[1])
        self.codebook_size = int(codebook_size)
        rows = [tuple(int(x) for x in row) for row in codes.tolist()]
        if len(set(rows)) != len(rows):
            raise ValueError("catalog-constrained item decoding requires injective SIDs")

        # transitions[visible_mask][partial_tuple] contains flattened d*V+c
        # transitions. Building all 2^L partial views is cheap for L=4/5 and
        # avoids scanning the catalog inside every evaluation batch.
        transition_sets = [defaultdict(set) for _ in range(1 << self.n_digit)]
        full_mask = (1 << self.n_digit) - 1
        for code in rows:
            for visible_mask in range(full_mask):
                partial = tuple(
                    code[d] if visible_mask & (1 << d) else -1
                    for d in range(self.n_digit)
                )
                bucket = transition_sets[visible_mask][partial]
                for d in range(self.n_digit):
                    if not visible_mask & (1 << d):
                        bucket.add(d * self.codebook_size + code[d])

        self.transitions = []
        for by_state in transition_sets:
            self.transitions.append(
                {
                    state: torch.tensor(sorted(values), dtype=torch.long)
                    for state, values in by_state.items()
                }
            )
        self.catalog = frozenset(rows)

    def allowed(self, state):
        state_tuple = tuple(int(x) for x in state)
        visible_mask = 0
        for d, value in enumerate(state_tuple):
            if value >= 0:
                visible_mask |= 1 << d
        return self.transitions[visible_mask].get(state_tuple)


def _merge_partial_states(child_states, child_scores, beam_act: int):
    """Log-sum-exp merge identical states and retain the strongest beams."""
    unique_states, inverse = torch.unique(
        child_states, dim=0, sorted=False, return_inverse=True
    )
    n_unique = unique_states.shape[0]
    maxima = child_scores.new_full((n_unique,), float("-inf"))
    maxima.scatter_reduce_(0, inverse, child_scores, reduce="amax", include_self=True)
    exp_sums = child_scores.new_zeros((n_unique,))
    exp_sums.scatter_add_(0, inverse, torch.exp(child_scores - maxima[inverse]))
    merged_scores = maxima + torch.log(exp_sums)

    keep = min(int(beam_act), n_unique)
    best_scores, best_indices = torch.topk(merged_scores, k=keep)
    return unique_states[best_indices], best_scores


def _merge_partial_states_max(child_states, child_scores, beam_act: int):
    """Keep the strongest route for duplicate partial states."""
    unique_states, inverse = torch.unique(
        child_states, dim=0, sorted=False, return_inverse=True
    )
    maxima = child_scores.new_full((unique_states.shape[0],), float("-inf"))
    maxima.scatter_reduce_(0, inverse, child_scores, reduce="amax", include_self=True)
    keep = min(int(beam_act), unique_states.shape[0])
    best_scores, best_indices = torch.topk(maxima, k=keep)
    return unique_states[best_indices], best_scores


def _catalog_constraint(model, tokenizer, vocab_size):
    constraint = getattr(model, "_catalog_constraint_cache", None)
    if constraint is None:
        catalog_codes = []
        for token_ids in tokenizer.item2tokens.values():
            catalog_codes.append([
                int(token_ids[d]) - (tokenizer.sid_offset + d * vocab_size)
                for d in range(model.n_digit)
            ])
        constraint = CatalogConstraint(catalog_codes, vocab_size)
        model._catalog_constraint_cache = constraint
    return constraint


@torch.no_grad()
def catalog_order_marginal_decode(
    model,
    encoder_hidden,
    tokenizer,
    n_return_sequences=10,
):
    """Beam search over catalog-valid partial SIDs with reveal-order merging."""
    device = encoder_hidden.device
    batch_size = encoder_hidden.shape[0]
    n_digit = int(model.n_digit)
    vocab_size = int(model.codebook_size)

    beam_cfg = model.config.get("vectorized_beam_search", {}) or {}
    split = model.config.get("current_split", "val")
    split_cfg = beam_cfg.get(split, beam_cfg)
    beam_act = int(split_cfg.get("beam_act", beam_cfg.get("beam_act", 32)))
    top_k = min(
        int(n_return_sequences),
        int(beam_cfg.get("top_k_final", n_return_sequences)),
    )
    catalog_cfg = model.config.get("catalog_beam", {}) or {}
    # A complete L-digit SID can be reached by L! reveal orders. Retaining at
    # least that many candidates per output beam prevents premature removal of
    # most alternate-order contributions before the state merge.
    configured_merge_factor = catalog_cfg.get("premerge_factor")
    merge_factor = int(
        math.factorial(n_digit)
        if configured_merge_factor is None
        else configured_merge_factor
    )
    merge_factor = max(1, merge_factor)

    constraint = _catalog_constraint(model, tokenizer, vocab_size)

    # Keep one beam tensor per example because catalog branching/merging can
    # produce a different number of live states for each example.
    states = [torch.full((1, n_digit), -1, dtype=torch.long, device=device)
              for _ in range(batch_size)]
    scores = [torch.zeros(1, device=device) for _ in range(batch_size)]

    for _ in range(n_digit):
        counts = [x.shape[0] for x in states]
        flat_states = torch.cat(states, dim=0)
        flat_encoder = torch.cat([
            encoder_hidden[b:b + 1].expand(counts[b], -1, -1)
            for b in range(batch_size)
        ], dim=0)
        decoder_input = torch.clamp(flat_states, min=0)
        mask_positions = flat_states.lt(0)
        outputs = model.forward_decoder_only({
            "decoder_input_ids": decoder_input,
            "encoder_hidden": flat_encoder,
            "mask_positions": mask_positions,
        }, digit=None, use_cache=False)
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

        next_states = []
        next_scores = []
        offset = 0
        for b, count in enumerate(counts):
            parent_states = flat_states[offset:offset + count]
            parent_logp = log_probs[offset:offset + count]
            legal_mask = torch.zeros(
                (count, n_digit * vocab_size), dtype=torch.bool, device=device
            )
            for parent in range(count):
                allowed = constraint.allowed(parent_states[parent].tolist())
                if allowed is not None:
                    legal_mask[parent, allowed.to(device)] = True

            candidate_scores = (
                scores[b][:, None] + parent_logp.reshape(count, -1)
            ).masked_fill(~legal_mask, float("-inf"))
            finite_count = int(torch.isfinite(candidate_scores).sum().item())
            if finite_count == 0:
                raise RuntimeError("catalog beam reached a state with no legal transition")
            premerge_k = min(finite_count, beam_act * merge_factor)
            best_scores, flat_indices = torch.topk(
                candidate_scores.reshape(-1), k=premerge_k
            )
            parent_indices = flat_indices // (n_digit * vocab_size)
            transition = flat_indices % (n_digit * vocab_size)
            digit_indices = transition // vocab_size
            token_indices = transition % vocab_size
            children = parent_states[parent_indices].clone()
            children[
                torch.arange(premerge_k, device=device), digit_indices
            ] = token_indices
            merged_states, merged_scores = _merge_partial_states(
                children, best_scores, beam_act
            )
            next_states.append(merged_states)
            next_scores.append(merged_scores)
            offset += count
        states, scores = next_states, next_scores

    outputs = []
    for b in range(batch_size):
        keep = min(top_k, scores[b].numel())
        _, order = torch.topk(scores[b], k=keep)
        result = states[b][order]
        if keep < top_k:
            result = torch.cat([result, result[-1:].expand(top_k - keep, -1)], dim=0)
        outputs.append(result)
    generated = torch.stack(outputs, dim=0)

    # This is both a safety assertion and a guard against accidental future
    # changes that turn catalog decoding back into SID-only evaluation.
    for row in generated.detach().cpu().reshape(-1, n_digit).tolist():
        if tuple(int(x) for x in row) not in constraint.catalog:
            raise RuntimeError(f"catalog decoder emitted illegal SID: {row}")
    return generated


@torch.no_grad()
def catalog_uncertainty_decode(
    model,
    encoder_hidden,
    tokenizer,
    n_return_sequences=10,
    return_scores=False,
):
    """Catalog-constrained beam search with a personalized reveal order.

    For every live partial SID, select the still-masked digit whose legal token
    distribution has the largest top-1/top-2 probability margin, then expand
    only that digit.  Consequently different examples and different live beams
    may follow different reveal orders, while every intermediate state remains
    extendable to a concrete catalog item.
    """
    device = encoder_hidden.device
    batch_size = encoder_hidden.shape[0]
    n_digit = int(model.n_digit)
    vocab_size = int(model.codebook_size)
    beam_cfg = model.config.get("vectorized_beam_search", {}) or {}
    split = model.config.get("current_split", "val")
    split_cfg = beam_cfg.get(split, beam_cfg)
    beam_act = int(split_cfg.get("beam_act", beam_cfg.get("beam_act", 32)))
    top_k = min(
        int(n_return_sequences),
        int(beam_cfg.get("top_k_final", n_return_sequences)),
    )
    beam_act = max(beam_act, top_k)
    constraint = _catalog_constraint(model, tokenizer, vocab_size)

    states = [
        torch.full((1, n_digit), -1, dtype=torch.long, device=device)
        for _ in range(batch_size)
    ]
    scores = [torch.zeros(1, device=device) for _ in range(batch_size)]

    for _ in range(n_digit):
        counts = [state.shape[0] for state in states]
        flat_states = torch.cat(states, dim=0)
        flat_encoder = torch.cat([
            encoder_hidden[b:b + 1].expand(counts[b], -1, -1)
            for b in range(batch_size)
        ], dim=0)
        outputs = model.forward_decoder_only({
            "decoder_input_ids": torch.clamp(flat_states, min=0),
            "encoder_hidden": flat_encoder,
            "mask_positions": flat_states.lt(0),
        }, digit=None, use_cache=False)
        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

        next_states = []
        next_scores = []
        offset = 0
        for batch_idx, count in enumerate(counts):
            batch_children = []
            batch_child_scores = []
            for parent_idx in range(count):
                state = flat_states[offset + parent_idx]
                allowed = constraint.allowed(state.tolist())
                if allowed is None or allowed.numel() == 0:
                    raise RuntimeError("uncertainty beam reached a state with no legal transition")
                allowed = allowed.to(device)
                allowed_digits = allowed // vocab_size
                allowed_codes = allowed % vocab_size

                best_digit = None
                best_key = None
                for digit in torch.unique(allowed_digits, sorted=True).tolist():
                    digit = int(digit)
                    codes = allowed_codes[allowed_digits == digit]
                    legal_logits = log_probs[offset + parent_idx, digit, codes]
                    legal_probs = torch.softmax(legal_logits, dim=0)
                    if legal_probs.numel() == 1:
                        margin = 1.0
                        top_probability = 1.0
                    else:
                        top_two = torch.topk(legal_probs, k=2).values
                        margin = float((top_two[0] - top_two[1]).item())
                        top_probability = float(top_two[0].item())
                    # Stable lower-digit tie break after margin and peak mass.
                    key = (margin, top_probability, -digit)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_digit = digit

                digit_codes = allowed_codes[allowed_digits == best_digit]
                children = state.unsqueeze(0).expand(digit_codes.numel(), -1).clone()
                children[:, best_digit] = digit_codes
                child_scores = (
                    scores[batch_idx][parent_idx]
                    + log_probs[offset + parent_idx, best_digit, digit_codes]
                )
                batch_children.append(children)
                batch_child_scores.append(child_scores)

            merged_states, merged_scores = _merge_partial_states_max(
                torch.cat(batch_children, dim=0),
                torch.cat(batch_child_scores, dim=0),
                beam_act,
            )
            next_states.append(merged_states)
            next_scores.append(merged_scores)
            offset += count
        states, scores = next_states, next_scores

    generated_rows = []
    generated_scores = []
    for batch_idx in range(batch_size):
        keep = min(top_k, scores[batch_idx].numel())
        best_scores, order = torch.topk(scores[batch_idx], k=keep)
        result = states[batch_idx][order]
        if keep < top_k:
            pad_count = top_k - keep
            result = torch.cat([result, result[-1:].expand(pad_count, -1)], dim=0)
            best_scores = torch.cat([
                best_scores,
                best_scores.new_full((pad_count,), float("-inf")),
            ])
        generated_rows.append(result)
        generated_scores.append(best_scores)

    generated = torch.stack(generated_rows, dim=0)
    path_scores = torch.stack(generated_scores, dim=0)
    for row in generated.detach().cpu().reshape(-1, n_digit).tolist():
        if tuple(int(value) for value in row) not in constraint.catalog:
            raise RuntimeError(f"uncertainty decoder emitted illegal SID: {row}")
    if return_scores:
        return generated, path_scores
    return generated
