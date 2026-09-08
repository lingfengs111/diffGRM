"""Parameter-matched encoder-only control for the one-pass SID drafter.

This model deliberately contains no masked/diffusion decoder.  It encodes the
observed item sequence with causal self-attention, pools the last valid history
position, and maps that representation to four parallel coordinate states.
The catalog item loss and pair/triple selector are supplied by the same
training code used for the masked-decoder drafter.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .model import EncoderBlock, ModelOutput, make_norm


class EncoderOnlyFourHeadDrafter(nn.Module):
    """SASRec-style history encoder followed by four parallel SID heads."""

    def __init__(self, config, dataset, tokenizer):
        super().__init__()
        del dataset  # The shared tokenizer owns all catalog metadata.
        self.config = config
        self.tokenizer = tokenizer
        self.n_digit = int(config['n_digit'])
        self.codebook_size = int(config['codebook_size'])
        self.vocab_size = int(tokenizer.vocab_size)
        self.n_embd = int(config['n_embd'])
        self.n_head = int(config['n_head'])
        self.n_inner = int(config['n_inner'])
        self.encoder_n_layer = int(config.get('encoder_head_n_layer', 4))
        self.max_history_len = int(config.get('max_history_len', 20))
        self.dropout = float(config.get('dropout', 0.1))
        self.normalize_logits = bool(
            config.get('encoder_head_normalize_logits', False)
        )
        self.logit_temperature = float(
            config.get('encoder_head_logit_temperature', 1.0)
        )
        if self.logit_temperature <= 0:
            raise ValueError('encoder-head logit temperature must be positive')
        norm_type = str(config.get('norm_type', 'layernorm'))
        norm_eps = float(config.get('norm_eps', 1e-5))

        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.n_embd, self.n_embd),
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.n_embd)
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(
                self.n_embd,
                self.n_head,
                self.n_inner,
                config['attn_pdrop'],
                config['resid_pdrop'],
                act='gelu',
                norm_type=norm_type,
                norm_eps=norm_eps,
            )
            for _ in range(self.encoder_n_layer)
        )
        self.encoder_norm = make_norm(norm_type, self.n_embd, norm_eps)
        self.coordinate_projections = nn.ModuleList(
            nn.Linear(self.n_embd, self.n_embd) for _ in range(self.n_digit)
        )
        self.coordinate_queries = nn.Parameter(
            torch.empty(self.n_digit, self.n_embd)
        )
        # A shared position-wise FFN makes this control parameter-matched to
        # the canonical 2-encoder/2-decoder DiffGRM backbone (within 0.03%).
        self.coordinate_mixer = nn.Sequential(
            nn.Linear(self.n_embd, self.n_inner),
            nn.GELU(),
            nn.Linear(self.n_inner, self.n_embd),
        )
        self.coordinate_norm = make_norm(norm_type, self.n_embd, norm_eps)
        self.output_adapter = nn.Identity()
        self.drop = nn.Dropout(self.dropout)
        self.apply(self._init_weights)
        nn.init.normal_(self.coordinate_queries, mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
            if getattr(module, 'weight', None) is not None:
                nn.init.ones_(module.weight)

    def _compute_digit_logits(self, hidden, digit):
        start = self.tokenizer.sid_offset + digit * self.codebook_size
        code_embeddings = self.embedding.weight[start:start + self.codebook_size]
        hidden = self.output_adapter(hidden)
        if self.normalize_logits:
            hidden = F.normalize(hidden, dim=-1)
            code_embeddings = F.normalize(code_embeddings, dim=-1)
        return hidden @ code_embeddings.t() / self.logit_temperature

    def forward(self, batch, return_loss=False):
        if return_loss:
            raise ValueError(
                'EncoderOnlyFourHeadDrafter is trained through the shared '
                'catalog/token objectives, not its forward method.'
            )
        device = next(self.parameters()).device
        history_sid = batch['history_sid'].to(device)
        valid = history_sid.ne(-1).any(dim=-1)
        batch_size, seq_len, n_digit = history_sid.shape
        if n_digit != self.n_digit:
            raise ValueError('history SID width does not match n_digit')

        history_tokens = torch.zeros_like(history_sid, dtype=torch.long)
        for digit in range(self.n_digit):
            codes = history_sid[:, :, digit]
            history_tokens[:, :, digit] = torch.where(
                codes.eq(-1),
                torch.zeros_like(codes),
                codes + self.tokenizer.sid_offset + digit * self.codebook_size,
            )
        history_tokens.clamp_(0, self.vocab_size - 1)
        token_embeddings = self.embedding(history_tokens)
        item_embeddings = self.item_mlp(
            token_embeddings.reshape(
                batch_size, seq_len, self.n_digit * self.n_embd
            )
        )
        positions = torch.arange(seq_len, device=device)
        hidden = self.drop(item_embeddings + self.pos_emb_enc(positions)[None])

        # Standard SASRec causality over the observed history.  Padding is on
        # the right in the shared tokenizer, so the last valid state is the
        # next-item query and can attend to every earlier observed item.
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        attention_mask = (
            causal[None, None, :, :] & valid[:, None, None, :]
        )
        for block in self.encoder_blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        hidden = self.encoder_norm(hidden)

        last_index = valid.long().sum(dim=1).sub(1).clamp_min(0)
        pooled = hidden[
            torch.arange(batch_size, device=device), last_index
        ]
        output = ModelOutput()
        # The shared one-pass interface passes this compact history state to
        # forward_decoder_only; retaining a length axis keeps its contract.
        output.hidden_states = pooled[:, None, :]
        if getattr(self, 'uses_full_history', False):
            output.hidden_states = hidden
            output.history_mask = valid
        return output

    def forward_decoder_only(
        self,
        batch,
        return_loss=False,
        digit=None,
        past_key_values=None,
        use_cache=False,
    ):
        del return_loss, digit, past_key_values, use_cache
        encoder_hidden = batch['encoder_hidden']
        if encoder_hidden.ndim != 3 or encoder_hidden.shape[1] != 1:
            raise ValueError('encoder-only drafter expects pooled [B,1,H] history')
        pooled = encoder_hidden[:, 0]
        coordinate_hidden = torch.stack(
            [projection(pooled) for projection in self.coordinate_projections],
            dim=1,
        )
        coordinate_hidden = F.gelu(
            coordinate_hidden + self.coordinate_queries[None]
        )
        coordinate_hidden = self.coordinate_norm(
            coordinate_hidden + self.coordinate_mixer(coordinate_hidden)
        )
        output = ModelOutput()
        output.hidden_states = coordinate_hidden
        output.logits = torch.stack(
            [
                self._compute_digit_logits(coordinate_hidden[:, digit], digit)
                for digit in range(self.n_digit)
            ],
            dim=1,
        )
        return output

    @property
    def n_parameters(self):
        return f"{sum(parameter.numel() for parameter in self.parameters()):,}"
