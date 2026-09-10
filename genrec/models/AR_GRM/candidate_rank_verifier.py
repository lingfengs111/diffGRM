"""Matched causal/bidirectional whole-SID ranking from one AR checkpoint.

The history encoder and embeddings stay frozen. Ranking reads the last SID
coordinate after processing [BOS, c0, ..., cD]. Only the candidate self-attention
mask differs between the two arms. Token supervision ALWAYS uses a separate
causal pass; visible candidate codes must never serve as their own LM targets.
"""

import torch
from torch import nn
import torch.nn.functional as F


class CandidateRankVerifier(nn.Module):
    def __init__(self, ar_model, attention="causal", zero_init_head=False):
        super().__init__()
        if attention not in ("causal", "bidirectional"):
            raise ValueError("attention must be causal or bidirectional")
        if not ar_model.use_causal_mask:
            raise ValueError("the auxiliary AR model must retain its causal mask")
        if getattr(ar_model.tokenizer, "sid_prefix_strategy", "none") != "none":
            raise ValueError("this experiment requires fixed, same-view SIDs")
        self.ar = ar_model
        self.attention = attention
        self.head = nn.Sequential(
            nn.Linear(ar_model.n_embd, ar_model.n_embd),
            nn.GELU(),
            nn.Linear(ar_model.n_embd, 1),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        if zero_init_head:
            # A residual verifier starts as the exact frozen AR verifier. The
            # first update learns only the final projection; subsequent updates
            # train the whole residual MLP without losing that safety property.
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        self.adapt_decoder = False
        self.set_decoder_trainable(False)

    def set_decoder_trainable(self, enabled):
        self.adapt_decoder = bool(enabled)
        self.ar.requires_grad_(False)
        self.ar.decoder_blocks.requires_grad_(self.adapt_decoder)
        self.head.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen history encoder (including shared final norm and
        # input dropout) deterministic throughout both training stages.
        self.ar.eval()
        self.ar.decoder_blocks.train(mode and self.adapt_decoder)
        self.head.train(mode)
        return self

    @torch.no_grad()
    def encode_history(self, history_sid):
        return self.ar({"history_sid": history_sid}, return_loss=False).hidden_states

    def candidate_hidden(self, encoded, history_mask, paths, cross_kv, width):
        ar = self.ar
        offsets = ar.tokenizer.sid_offset + torch.arange(
            ar.n_digit, device=paths.device
        ) * ar.codebook_size
        embedded = ar.embedding(paths + offsets)
        hidden = torch.cat([
            ar.bos_embedding[None, None].expand(paths.shape[0], 1, -1),
            embedded,
        ], dim=1)
        mask = ar._causal_mask(paths.shape[0], hidden.shape[1], hidden.device) \
            if self.attention == "causal" else None
        key_mask = history_mask[:, None, None, :].repeat_interleave(width, 0)
        for block, kv in zip(ar.decoder_blocks, cross_kv):
            hidden = block(
                hidden, encoder_hidden=None, attention_mask=mask,
                cross_key_value=kv.repeat_interleave(width, 0),
                cross_attention_mask=key_mask,
            )["hidden_states"]
        return ar.ln_f(hidden)

    def score_encoded(self, encoded, history_mask, candidates, chunk_size=16):
        if candidates.ndim != 3 or candidates.shape[2] != self.ar.n_digit:
            raise ValueError("candidates must have shape [B,C,D]")
        if candidates.shape[0] != encoded.shape[0] or history_mask.shape != encoded.shape[:2]:
            raise ValueError("history and candidate dimensions do not match")
        if candidates.shape[1] < 1 or chunk_size < 1:
            raise ValueError("candidate count and chunk size must be positive")
        batch_size, count, digits = candidates.shape
        # ar.training is deliberately False for its frozen history. Do not
        # infer detach from it: decoder cross-attention K/V need rank gradients.
        cross_kv = self.ar._precompute_cross_kv(encoded, detach=False)
        chunks = []
        for start in range(0, count, chunk_size):
            paths = candidates[:, start:start + chunk_size]
            width = paths.shape[1]
            hidden = self.candidate_hidden(
                encoded, history_mask, paths.reshape(-1, digits), cross_kv, width
            )
            chunks.append(self.head(hidden[:, -1]).reshape(batch_size, width))
        return torch.cat(chunks, dim=1)

    def causal_token_loss(self, encoded, history_mask, targets):
        kv = self.ar._precompute_cross_kv(encoded, detach=False)
        logits = self.ar._candidate_logits_from_encoded_history(
            encoded, history_mask[:, None, None, :], kv, targets, 1
        )
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def forward(self, history_sid, candidates, chunk_size=16, targets=None):
        encoded = self.encode_history(history_sid)
        mask = history_sid.ne(-1).any(dim=-1)
        scores = self.score_encoded(encoded, mask, candidates, chunk_size)
        auxiliary = self.causal_token_loss(encoded, mask, targets) \
            if targets is not None else None
        return scores, auxiliary
