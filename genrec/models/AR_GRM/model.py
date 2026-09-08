# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

        # Initialize weights
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, attention_mask=None, key_value=None, past_key_value=None, use_cache=False, is_decoder_self_attn=False):
        B, T, C = x.size()

        if key_value is not None:
            # Cross attention: Q from x, K,V from key_value
            q = self.qkv(x)[:, :, :self.emb_dim]  # Only take Q part
            k, v = key_value.chunk(2, dim=-1)  # key_value should be [B, T_enc, 2*emb_dim]
            T_kv = k.size(1)
        else:
            # Self attention
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            T_kv = T

        # Handle past key-value cache for incremental decoding
        if past_key_value is not None and use_cache:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            T_kv = k.size(1)

        # 保存拼接后的完整k和v用于cache（在reshape之前）
        k_for_cache = k
        v_for_cache = v

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, T_kv)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T, T_kv) or (B, 1, T, T_kv)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            att = att.masked_fill(attention_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)  # 🚀 修复：防止全屏蔽行的 NaN 扩散
        
        # 🚀 改进：使用更稳健的全零归一化，避免全屏蔽行引入PAD信息泄露
        if attention_mask is not None:
            # 再次乘 mask 并做归一化，确保没有合法 key 时该行注意力全零
            att = att * attention_mask  # 广播到 (B, n_head, T, T_kv)
            denom = att.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            att = att / denom
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, emb_dim)

        # Output projection
        y = self.resid_dropout(self.proj(y))

        # Prepare cache for next iteration - 保存原始的3维k和v
        present_key_value = (k_for_cache, v_for_cache) if use_cache else None

        return y, present_key_value


class FeedForward(nn.Module):

    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, n_inner)
        self.c_proj = nn.Linear(n_inner, emb_dim)
        self.dropout = nn.Dropout(resid_drop)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', layer_norm_epsilon=1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接（非decoder自注意力）
        attn_output, _ = self.attn(self.ln_1(x), attention_mask=attention_mask, is_decoder_self_attn=False)
        x = x + attn_output
        
        # 前馈网络 + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', layer_norm_epsilon=1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_3 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, encoder_hidden=None, attention_mask=None, 
                past_key_value=None, use_cache=False, cross_key_value=None, cross_attention_mask=None):
        # 自注意力（支持传入因果掩码）
        self_past_kv = None
        cross_past_kv = None
        if past_key_value is not None:
            if len(past_key_value) >= 1:
                self_past_kv = past_key_value[0]
            if len(past_key_value) >= 2:
                cross_past_kv = past_key_value[1]
        
        attn_output, present_key_value = self.self_attn(
            self.ln_1(x), 
            attention_mask=attention_mask,  # 允许因果掩码
            past_key_value=self_past_kv,
            use_cache=use_cache,
            is_decoder_self_attn=True
        )
        x = x + attn_output

        # 交叉注意力
        if encoder_hidden is not None or cross_key_value is not None:
            if cross_key_value is not None:
                # 🚀 使用预计算的KV，避免重复计算
                encoder_kv = cross_key_value
            else:
                # 🚀 修复：使用与训练/推理一致的线性投影路径
                kv_proj = self.cross_attn.qkv(encoder_hidden)  # [B,S,3*D]
                D = self.cross_attn.emb_dim
                k = kv_proj[..., D:2*D]
                v = kv_proj[..., 2*D:]
                encoder_kv = torch.cat([k, v], dim=-1)  # [B,S,2*D]
            
            cross_attn_output, cross_present = self.cross_attn(
                self.ln_2(x),
                key_value=encoder_kv,
                past_key_value=cross_past_kv,
                use_cache=use_cache,
                attention_mask=cross_attention_mask  # 新增：cross attention mask
            )
            x = x + cross_attn_output
            
            if use_cache:
                present_key_value = (present_key_value, cross_present)
        
        # 前馈网络
        x = x + self.mlp(self.ln_3(x))
        
        return_dict = {}
        return_dict['hidden_states'] = x
        if use_cache:
            return_dict['present_key_value'] = present_key_value
        
        return return_dict


class ModelOutput:

    def __init__(self):
        self.loss = None
        self.logits = None
        self.hidden_states = None
        self.past_key_values = None


class AR_GRM(AbstractModel):

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super().__init__(config, dataset, tokenizer)
        
        self.config = config
        self.tokenizer = tokenizer
        self.n_digit = config['n_digit']
        self.codebook_size = config['codebook_size']
        self.vocab_size = tokenizer.vocab_size
        self.constrained_beam = bool(config.get('constrained_beam', True))
        self.filter_illegal_final = bool(config.get('filter_illegal_final', True))
        self._prefix_allowed = None
        if self.constrained_beam:
            self._prefix_allowed = self._build_prefix_allowed(tokenizer)
        
        # Model dimensions
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_inner = config['n_inner']
        self.dropout = config['dropout']
        
        # Encoder layers
        self.encoder_n_layer = config['encoder_n_layer']
        self.decoder_n_layer = config['decoder_n_layer']
        
        # 自回归特有设置
        self.use_causal_mask = bool(config.get('use_causal_mask', True))
        
        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # 添加与RPG_ED一致的item_mlp：将n_digit个SID token压缩为1个token
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.n_embd, self.n_embd),  # n_digit×d → d
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        # BOS embedding（用于自回归 decoder 起始）
        self.bos_embedding = nn.Parameter(torch.randn(self.n_embd) * 0.02)
        
        # 位置编码：只为encoder添加绝对位置编码（与RPG_ED一致）
        self.max_history_len = config.get('max_history_len', 20)  # Amazon 默认协议
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.n_embd)
        # 移除decoder位置编码，decoder只使用掩码
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop']
            )
            for _ in range(self.encoder_n_layer)
        ])
        
        # Decoder blocks  
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop']
            )
            for _ in range(self.decoder_n_layer)
        ])
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(self.n_embd)
        
        # -- Decoder output parameterization --
        # ``tied_embedding`` is the historical implementation.  ``level_linear``
        # follows GRID's level-wise output design: each SID position gets its own
        # classifier rather than reusing the input-token embedding slice.
        self.decoder_output_mode = str(
            self.config.get('decoder_output_mode', 'tied_embedding')
        ).lower()
        if self.decoder_output_mode not in ('tied_embedding', 'level_linear'):
            raise ValueError(
                'decoder_output_mode must be tied_embedding|level_linear, '
                f'got {self.decoder_output_mode}'
            )

        share_out = self.config.get('share_decoder_output_embedding', True)
        if self.decoder_output_mode == 'level_linear':
            self.output_adapter = nn.Identity()
            self.level_output_heads = nn.ModuleList([
                nn.Linear(self.n_embd, self.codebook_size, bias=False)
                for _ in range(self.n_digit)
            ])
            print('[AR_GRM] Using one independent output head per SID level')
        elif share_out:
            # 直接 weight-tying，不新增参数
            self.output_adapter = nn.Identity()
            print(f"[AR_GRM] Using shared embedding dot-product output layer")
        else:
            # 若以后要回滚到独立 head，用这一行
            self.output_adapter = nn.Linear(self.n_embd, self.n_embd, bias=False)
            print(f"[AR_GRM] Using independent Linear output adapter")
        # -------------------------------------------------------------
        
        # Dropout
        self.drop = nn.Dropout(self.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        if self.decoder_output_mode == 'level_linear':
            self.initialize_level_output_heads_from_embedding()
            if bool(self.config.get('train_level_output_heads_only', False)):
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.level_output_heads.parameters():
                    parameter.requires_grad_(True)
                print('[AR_GRM] Training only the level-specific output heads')

    @torch.no_grad()
    def initialize_level_output_heads_from_embedding(self):
        """Make level heads exactly reproduce tied-embedding logits at init."""
        if self.decoder_output_mode != 'level_linear':
            return
        for digit, head in enumerate(self.level_output_heads):
            start = self.tokenizer.sid_offset + digit * self.codebook_size
            end = start + self.codebook_size
            head.weight.copy_(self.embedding.weight[start:end])

    def load_init_checkpoint(self, state_dict):
        """Warm-start a level-head model from a historical tied-head checkpoint."""
        if self.decoder_output_mode != 'level_linear':
            self.load_state_dict(state_dict)
            return

        incompatible = self.load_state_dict(state_dict, strict=False)
        expected_missing = {
            f'level_output_heads.{digit}.weight'
            for digit in range(self.n_digit)
        }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != expected_missing or unexpected:
            raise RuntimeError(
                'Unexpected keys while warm-starting level heads: '
                f'missing={sorted(missing)}, unexpected={sorted(unexpected)}'
            )
        # Copy *loaded* embedding weights, not the random construction-time
        # weights, so the first forward pass is an exact tied-head control.
        self.initialize_level_output_heads_from_embedding()

    def _compute_digit_logits(self, hidden_last, digit):
        """
        使用共享embedding的dot-product计算logits
        
        Args:
            hidden_last: (B, d_model) - decoder输出的隐藏状态
            digit: 0..n_digit-1 - 要预测的digit位置
            
        Returns:
            logits: (B, codebook_size) - 预测logits
        """
        if digit is None:
            raise ValueError("digit参数不能为None，必须指定要计算的codebook位置")
        
        if digit >= self.n_digit:
            raise ValueError(f"digit={digit} 超出范围，应该在 [0, {self.n_digit-1}]")
        
        if self.decoder_output_mode == 'level_linear':
            return self.level_output_heads[digit](hidden_last)

        # 2.1 取出 embedding matrix 的相应切片
        # token ID 布局 = [PAD, BOS, EOS, digit0 256 个, digit1 256 个, ...]
        start = self.tokenizer.sid_offset + digit * self.codebook_size
        end = start + self.codebook_size  # 不含 end
        # shape: (codebook_size, d_model)
        E_sub = self.embedding.weight[start:end]
        
        # 2.2 optional adapter
        h = self.output_adapter(hidden_last)  # (B, d_model)
        
        # 2.3 dot-product 得 logits
        # (B, d_model) @ (d_model, codebook_size).T → (B, codebook_size)
        logits = torch.matmul(h, E_sub.t())
        
        return logits

    @property
    def n_parameters(self) -> str:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params:,}"

    def _causal_mask(self, B: int, T: int, device):
        mask = torch.tril(torch.ones(T, T, device=device))
        return mask.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

    def _build_prefix_allowed(self, tokenizer):
        """Build a fixed-order catalog trie over raw codebook IDs."""
        allowed = [dict() for _ in range(self.n_digit)]
        for item, tokens in tokenizer.item2tokens.items():
            for token_path in tokenizer.iter_item_token_paths(item, tokens):
                path = tuple(
                    int(token) - (tokenizer.sid_offset + digit * self.codebook_size)
                    for digit, token in enumerate(token_path)
                )
                for digit in range(self.n_digit):
                    prefix = path[:digit]
                    allowed[digit].setdefault(prefix, set()).add(path[digit])
        return [
            {prefix: tuple(sorted(values)) for prefix, values in level.items()}
            for level in allowed
        ]

    def _constrain_log_probs(self, log_probs, prefixes, digit):
        """Mask token expansions that are absent from the catalog trie."""
        if not self.constrained_beam:
            return log_probs
        if digit == 0:
            valid = self._prefix_allowed[0].get((), ())
            mask = torch.zeros(self.codebook_size, dtype=torch.bool, device=log_probs.device)
            if valid:
                mask[list(valid)] = True
            return log_probs.masked_fill(~mask.unsqueeze(0), float('-inf'))

        if prefixes is None or prefixes.shape[1] != digit:
            raise ValueError(f"digit {digit} requires prefixes [N,{digit}]")
        # Prefix lookup is catalog metadata, not model computation.  Building
        # one dense boolean mask per decoding step keeps the probability path
        # fully vectorized after this small CPU-side lookup.
        allowed_mask = torch.zeros(
            prefixes.shape[0], self.codebook_size, dtype=torch.bool
        )
        for row_idx, prefix in enumerate(prefixes.detach().cpu().tolist()):
            values = self._prefix_allowed[digit].get(tuple(prefix), ())
            if values:
                allowed_mask[row_idx, list(values)] = True
        return log_probs.masked_fill(~allowed_mask.to(log_probs.device), float('-inf'))

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        """
        自回归训练：teacher forcing（输入=[BOS, sid0..sid3]，在位置0..3预测sid0..3）
        """
        device = next(self.parameters()).device
        
        if not hasattr(self, '_debug_printed'):
            print(f"[AR_GRM] Encoder: MLP compression + abs pos encoding")
            print(f"[AR_GRM] vocab_size: {self.vocab_size}, codebook_size: {self.codebook_size}")
            self._debug_printed = True
        
        # --- Encoder ---
        history_sid = batch['history_sid'].to(device)  # [B, seq_len, n_digit]
        B, seq_len, n_digit = history_sid.shape
        
        # The shared tokenizer represents history as raw codebook IDs and uses
        # -1 (not code 0) as padding, matching the diffusion model exactly.
        valid_k = history_sid.ne(-1).any(dim=-1)            # [B, S] True=有效位置
        # 🚀 设计说明：使用 (B,1,1,S) 形状，表示"只屏蔽 Key 端"
        # 这会广播到 (B,n_head,T,S)，让注意力机制不看 PAD 位置的 key
        enc_key_mask = valid_k[:, None, None, :]            # [B,1,1,S] 只屏蔽 Key 端
        
        history_tokens = torch.zeros_like(history_sid, dtype=torch.long)
        for digit in range(n_digit):
            raw_codes = history_sid[:, :, digit]
            history_tokens[:, :, digit] = torch.where(
                raw_codes.eq(-1),
                torch.zeros_like(raw_codes),
                raw_codes + self.tokenizer.sid_offset + digit * self.codebook_size,
            )
        history_tokens = history_tokens.clamp(0, self.vocab_size - 1)
        
        # 2. 获取token嵌入
        tok_emb = self.embedding(history_tokens)  # [B, seq_len, n_digit, d]
        B, S, _, d = tok_emb.shape
        
        # 3. 重塑并通过MLP压缩：n_digit个SID token → 1个item token
        item_emb = tok_emb.reshape(B, S, self.n_digit * d)  # [B, S, n_digit*d]
        item_emb = self.item_mlp(item_emb)  # [B, S, d]
        
        # 4. 添加位置编码（与RPG_ED一致）
        pos_ids = torch.arange(S, device=item_emb.device)  # (S,)
        pos_emb = self.pos_emb_enc(pos_ids)  # (S, d)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)
        
        # 5. 将位置编码加到item_emb上
        encoder_hidden = item_emb + pos_emb  # [B, S, d]
        encoder_hidden = self.drop(encoder_hidden)
        
        # Pass through encoder blocks with padding mask
        for block in self.encoder_blocks:
            # 传入 enc_key_mask，让 encoder 不看 padding 的 key
            encoder_hidden = block(encoder_hidden, attention_mask=enc_key_mask)
        
        encoder_hidden = self.ln_f(encoder_hidden)  # [B, S, d]

        if not return_loss:
            out = ModelOutput()
            out.hidden_states = encoder_hidden
            return out

        # Teacher forcing inputs
        dec_gt = torch.clamp(batch['decoder_input_ids'].to(device), 0, self.codebook_size - 1)  # [B, n_digit]
        # 🚀 修复：保留 -100 原样，避免误训练为类别 0
        # 备选方案：若担心脏数据，可用以下代码保留 -100，其它非负再 clamp
        # _raw = batch['decoder_labels'].to(device)
        # labels = torch.where(_raw >= 0, _raw.clamp(0, self.codebook_size - 1), _raw)
        labels = batch['decoder_labels'].to(device)  # 保留 -100 用于 ignore_index
        B_dec = dec_gt.size(0)  # 🚀 修复：避免变量名冲突，提升可读性

        # Convert codebook id -> token id per digit
        token_ids = []
        for d in range(self.n_digit):
            tok = dec_gt[:, d] + self.tokenizer.sid_offset + d * self.codebook_size
            tok = torch.clamp(tok, 0, self.vocab_size - 1)
            token_ids.append(tok)
        token_ids = torch.stack(token_ids, dim=1)  # [B, n_digit]

        tok_emb = self.embedding(token_ids)  # [B, n_digit, d]
        bos = self.bos_embedding.unsqueeze(0).unsqueeze(1).expand(B_dec, 1, -1)  # [B,1,d]
        dec_inp = torch.cat([bos, tok_emb], dim=1)  # [B, n_digit+1, d]
        dec_inp = self.drop(dec_inp)

        # Decoder with causal mask and cross-attn
        x = dec_inp
        attn_mask = self._causal_mask(B_dec, x.size(1), device) if self.use_causal_mask else None
        # 预计算 cross-KV（每层一次）
        encoder_kv_list = []
        for blk in self.decoder_blocks:
            kv_proj = blk.cross_attn.qkv(encoder_hidden)
            k = kv_proj[..., self.n_embd:2*self.n_embd]
            v = kv_proj[..., 2*self.n_embd:]
            encoder_kv_list.append(torch.cat([k, v], dim=-1))

        for i, blk in enumerate(self.decoder_blocks):
            # 给 self-attn 传因果 mask；给 cross-attn 也传 enc_key_mask
            out = blk(
                x, 
                encoder_hidden=encoder_hidden, 
                attention_mask=attn_mask,                 # 自注意力的因果mask
                past_key_value=None, 
                use_cache=False, 
                cross_key_value=encoder_kv_list[i],
                cross_attention_mask=enc_key_mask         # 新增：cross attention的padding mask
            )
            x = out['hidden_states']

        x = self.ln_f(x)  # [B, n_digit+1, d]

        # Compute losses at positions 0..n_digit-1 predicting sid0..sid(n-1)
        total_loss = 0.0
        digit_logits = []
        for d in range(self.n_digit):
            logits_d = self._compute_digit_logits(x[:, d, :], digit=d)
            digit_logits.append(logits_d)
            total_loss = total_loss + F.cross_entropy(
                logits_d, labels[:, d], reduction='mean',
                label_smoothing=self.config.get('label_smoothing', 0.0),
                ignore_index=-100  # 🚀 修复：忽略未知商品的-100标签
            )
        total_loss = total_loss / self.n_digit

        out = ModelOutput()
        out.loss = total_loss
        # Expose teacher-forced conditionals for decoder diagnostics and exact
        # candidate-path verification.  This adds no training computation.
        out.logits = torch.stack(digit_logits, dim=1)
        out.hidden_states = x
        return out

    def _precompute_cross_kv(self, encoder_hidden, detach=True):
        """Project encoder states once for cached candidate scoring.

        Inference callers detach this cache.  Proposal-aware verifier
        fine-tuning must keep the projection graph, otherwise the listwise
        objective silently cannot update the decoder cross-attention K/V
        projections.
        """
        kv_list = []
        for blk in self.decoder_blocks:
            context = torch.no_grad() if detach else nullcontext()
            with context:
                qkv = blk.cross_attn.qkv(encoder_hidden)
                k = qkv[..., self.n_embd:2*self.n_embd]
                v = qkv[..., 2*self.n_embd:]
                kv_list.append(torch.cat([k, v], dim=-1))
        return kv_list

    def _score_candidate_paths_uncached(self, batch, candidate_codes, chunk_size):
        """Reference scorer that re-encodes history for every candidate chunk.

        Kept as a correctness oracle for the cached verifier.  Production
        inference should call :meth:`score_candidate_paths`, which encodes the
        shared history once.
        """
        device = next(self.parameters()).device
        batch_size, n_candidates, n_digit = candidate_codes.shape
        candidates = candidate_codes.to(device).long()
        score_chunks = []
        for start in range(0, n_candidates, chunk_size):
            end = min(start + chunk_size, n_candidates)
            width = end - start
            paths = candidates[:, start:end].reshape(batch_size * width, n_digit)
            expanded = {
                'history_sid': batch['history_sid'].to(device).repeat_interleave(
                    width, dim=0
                ),
                'decoder_input_ids': paths,
                'decoder_labels': paths,
            }
            if 'history_mask' in batch:
                expanded['history_mask'] = batch['history_mask'].to(
                    device
                ).repeat_interleave(width, dim=0)
            outputs = self.forward(expanded, return_loss=True)
            log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
            token_scores = log_probs.gather(-1, paths.unsqueeze(-1)).squeeze(-1)
            score_chunks.append(
                token_scores.sum(dim=-1).reshape(batch_size, width)
            )
        return torch.cat(score_chunks, dim=1)

    def _candidate_logits_from_encoded_history(
        self,
        encoder_hidden,
        enc_key_mask,
        cross_kv,
        paths,
        candidates_per_history,
    ):
        """Teacher-force candidate paths without re-running the history encoder."""
        expanded_mask = enc_key_mask.repeat_interleave(
            candidates_per_history, dim=0
        )
        expanded_cross_kv = [
            values.repeat_interleave(candidates_per_history, dim=0)
            for values in cross_kv
        ]

        token_ids = []
        for digit in range(self.n_digit):
            token_ids.append(
                paths[:, digit]
                + self.tokenizer.sid_offset
                + digit * self.codebook_size
            )
        token_ids = torch.stack(token_ids, dim=1).clamp(
            0, self.vocab_size - 1
        )
        token_embeddings = self.embedding(token_ids)
        bos = self.bos_embedding.view(1, 1, -1).expand(
            paths.shape[0], 1, -1
        )
        hidden = self.drop(torch.cat([bos, token_embeddings], dim=1))
        causal_mask = (
            self._causal_mask(
                paths.shape[0], hidden.shape[1], hidden.device
            )
            if self.use_causal_mask
            else None
        )

        for layer_idx, block in enumerate(self.decoder_blocks):
            output = block(
                hidden,
                # Cross-attention only needs the already projected K/V here;
                # do not materialize another [B*C,S,H] copy of encoder states.
                encoder_hidden=None,
                attention_mask=causal_mask,
                past_key_value=None,
                use_cache=False,
                cross_key_value=expanded_cross_kv[layer_idx],
                cross_attention_mask=expanded_mask,
            )
            hidden = output['hidden_states']
        hidden = self.ln_f(hidden)
        return torch.stack(
            [
                self._compute_digit_logits(hidden[:, digit, :], digit=digit)
                for digit in range(self.n_digit)
            ],
            dim=1,
        )

    def score_candidate_paths(
        self,
        batch,
        candidate_codes,
        chunk_size=None,
        use_cached_history=True,
    ):
        """Score complete item candidates with exact AR teacher forcing.

        Args:
            batch: history batch using the shared raw-code representation.
            candidate_codes: ``[B, C, L]`` legal raw codebook IDs.
            chunk_size: maximum number of candidates scored per forward pass.
            use_cached_history: encode history and project cross-attention K/V
                once for all candidate chunks.  ``False`` retains the original
                implementation for numerical-equivalence tests.

        Returns:
            Sum of digit log-probabilities with shape ``[B, C]``.  For a
            random-latent tokenizer, every legal latent prefix is scored and
            reduced with ``latent_item_aggregation``.  This matches Latte's
            item-level max/sum-over-paths semantics instead of accidentally
            verifying only the canonical latent-0 path.
        """
        if candidate_codes.ndim != 3:
            raise ValueError(
                f"candidate_codes must be [B,C,L], got {candidate_codes.shape}"
            )
        batch_size, n_candidates, n_digit = candidate_codes.shape
        if n_digit != self.n_digit:
            raise ValueError(f"candidate width {n_digit} != n_digit {self.n_digit}")
        if batch['history_sid'].shape[0] != batch_size:
            raise ValueError("candidate and history batch sizes do not match")

        device = next(self.parameters()).device
        candidates = candidate_codes.to(device).long()
        item_candidate_count = n_candidates
        latent_paths = 1
        if getattr(self.tokenizer, 'sid_prefix_strategy', 'none') == 'random_latent':
            latent_paths = int(self.config.get('n_latent_tokens', 8))
            candidates = candidates.unsqueeze(2).expand(
                batch_size, n_candidates, latent_paths, n_digit
            ).clone()
            candidates[:, :, :, 0] = torch.arange(
                latent_paths, device=device
            ).view(1, 1, latent_paths)
            candidates = candidates.reshape(
                batch_size, n_candidates * latent_paths, n_digit
            )
        chunk_size = int(
            chunk_size
            or self.config.get('candidate_score_chunk_size', 16)
        )
        if not use_cached_history:
            scores = self._score_candidate_paths_uncached(
                batch, candidates, chunk_size
            )
        else:
            history_sid = batch['history_sid'].to(device)
            encoder_hidden = self.forward(
                {'history_sid': history_sid}, return_loss=False
            ).hidden_states
            history_mask = history_sid.ne(-1).any(dim=-1)
            scores = self.score_candidate_paths_from_encoded_history(
                encoder_hidden,
                history_mask,
                candidates,
                chunk_size=chunk_size,
            )

        if latent_paths == 1:
            return scores
        scores = scores.reshape(batch_size, item_candidate_count, latent_paths)
        aggregation = str(
            self.config.get('latent_item_aggregation', 'logsumexp')
        ).lower()
        if aggregation == 'max':
            return scores.max(dim=-1).values
        if aggregation == 'logsumexp':
            return torch.logsumexp(scores, dim=-1)
        raise ValueError(
            'latent_item_aggregation must be logsumexp|max, '
            f'got {aggregation}'
        )

    def candidate_logits_from_encoded_history(
        self,
        encoder_hidden,
        history_mask,
        candidate_codes,
        chunk_size=None,
    ):
        """Teacher-force complete paths against externally encoded history.

        This is the shared-encoder interface: a proposal model may encode the
        observed sequence once, then both its catalog head and this AR decoder
        consume the same states.  The method deliberately retains gradients
        through the AR decoder and, when supplied, through ``encoder_hidden``.

        Returns logits with shape ``[B,C,D,K]`` for ``C`` complete candidate
        paths, ``D`` SID coordinates, and codebook size ``K``.
        """
        if encoder_hidden.ndim != 3:
            raise ValueError('encoder_hidden must have shape [B,S,H]')
        if candidate_codes.ndim != 3:
            raise ValueError('candidate_codes must have shape [B,C,D]')
        batch_size, n_candidates, n_digit = candidate_codes.shape
        if batch_size != encoder_hidden.shape[0]:
            raise ValueError('candidate and encoder batch sizes do not match')
        if n_digit != self.n_digit:
            raise ValueError(f'candidate width {n_digit} != n_digit {self.n_digit}')
        if encoder_hidden.shape[-1] != self.n_embd:
            raise ValueError(
                f'encoder width {encoder_hidden.shape[-1]} != n_embd {self.n_embd}'
            )

        device = next(self.parameters()).device
        encoder_hidden = encoder_hidden.to(device)
        candidates = candidate_codes.to(device).long()
        history_mask = history_mask.to(device).bool()
        if history_mask.shape != encoder_hidden.shape[:2]:
            raise ValueError('history_mask must have shape [B,S]')
        enc_key_mask = history_mask[:, None, None, :]
        chunk_size = int(
            chunk_size
            or self.config.get('candidate_score_chunk_size', 16)
        )
        cross_kv = self._precompute_cross_kv(
            encoder_hidden,
            detach=not self.training,
        )

        logit_chunks = []
        for start in range(0, n_candidates, chunk_size):
            end = min(start + chunk_size, n_candidates)
            width = end - start
            paths = candidates[:, start:end].reshape(batch_size * width, n_digit)
            logits = self._candidate_logits_from_encoded_history(
                encoder_hidden,
                enc_key_mask,
                cross_kv,
                paths,
                width,
            )
            logit_chunks.append(
                logits.reshape(
                    batch_size,
                    width,
                    self.n_digit,
                    self.codebook_size,
                )
            )
        return torch.cat(logit_chunks, dim=1)

    def score_candidate_paths_from_encoded_history(
        self,
        encoder_hidden,
        history_mask,
        candidate_codes,
        chunk_size=None,
    ):
        """Return exact AR path log-probabilities using shared history states."""
        candidates = candidate_codes.to(next(self.parameters()).device).long()
        logits = self.candidate_logits_from_encoded_history(
            encoder_hidden,
            history_mask,
            candidates,
            chunk_size=chunk_size,
        )
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_scores = log_probs.gather(
            -1, candidates.unsqueeze(-1)
        ).squeeze(-1)
        return token_scores.sum(dim=-1)

    def _decode_step(self, x_last, cross_kv_list, past_kv=None, enc_key_mask=None):
        # x_last: [N,1,d]
        x = x_last
        present = []
        for i, blk in enumerate(self.decoder_blocks):
            self_out, self_present = self.self_attend_step(blk, x, past_kv[i] if past_kv else None)
            x = x + self_out
            cross_out, cross_present = blk.cross_attn(
                blk.ln_2(x), 
                key_value=cross_kv_list[i], 
                use_cache=True,
                attention_mask=enc_key_mask   # 新增
            )
            x = x + cross_out
            x = x + blk.mlp(blk.ln_3(x))
            present.append((self_present, cross_present))
        x = self.ln_f(x)
        return x, present

    def self_attend_step(self, blk: DecoderBlock, x, past_kv_layer=None):
        # 单步自注意，query_len=1，无需显式掩码
        out, present = blk.self_attn(blk.ln_1(x), past_key_value=past_kv_layer, use_cache=True, is_decoder_self_attn=True)
        return out, present

    def generate(self, batch, n_return_sequences=10, mode=None):
        """顺序自回归 beam search，按 digit0→digit1→... 生成。
        返回 [B, top_k_final, n_digit]
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                enc_out = self.forward(batch, return_loss=False)
                encoder_hidden = enc_out.hidden_states
                B = encoder_hidden.size(0)
                cfg = self.config.get('ar_beam_search', {})
                split_cfg = cfg.get(self.config.get('current_split', 'val'), {})
                if split_cfg:
                    cfg = {**cfg, **split_cfg}
                pre_cut = cfg.get('pre_cut_num', [256]*self.n_digit)
                beam_num = cfg.get('beam_search_num', [256]*self.n_digit)
                TOPK = min(cfg.get('top_k_final', n_return_sequences), n_return_sequences)

                # 生成encoder的padding mask（与训练时一致）
                history_sid = batch['history_sid'].to(encoder_hidden.device)
                valid_k = history_sid.ne(-1).any(dim=-1)
                # 🚀 设计说明：使用 (B,1,1,S) 形状，表示"只屏蔽 Key 端"
                enc_key_mask = valid_k[:, None, None, :]

                # 预计算 cross-KV
                cross_kv = self._precompute_cross_kv(encoder_hidden)

                device = encoder_hidden.device
                bos = self.bos_embedding.to(device).unsqueeze(0).unsqueeze(1).expand(B, 1, -1)
                # step0
                x, present = self._decode_step(bos, cross_kv, past_kv=None, enc_key_mask=enc_key_mask)
                logits0 = self._compute_digit_logits(x[:, 0, :], digit=0)
                logp0 = F.log_softmax(logits0, dim=-1)
                logp0 = self._constrain_log_probs(logp0, prefixes=None, digit=0)

                topk_p0, topk_i0 = torch.topk(logp0, k=pre_cut[0], dim=-1)
                keep_k = min(beam_num[0], pre_cut[0])  # 防止步0的keep超过pre_cut
                best_lp, best_idx = torch.topk(topk_p0, k=keep_k, dim=-1)
                tok0 = topk_i0.gather(1, best_idx)

                # expand caches to beams
                def expand_to_beam(t, k):
                    return t.unsqueeze(1).repeat(1, k, *([1]*(t.ndim-1))).view(B*k, *t.shape[1:])

                beam_self_kv = []
                for l in range(len(self.decoder_blocks)):
                    self_k, cross_k = present[l]
                    if self_k is None:
                        beam_self_kv.append(None)
                    else:
                        k,v = self_k
                        beam_self_kv.append((expand_to_beam(k, keep_k), expand_to_beam(v, keep_k)))

                def emb_of_digit_token(digit, codebook_id):
                    token_id = codebook_id + self.tokenizer.sid_offset + digit * self.codebook_size
                    token_id = torch.clamp(token_id, 0, self.vocab_size - 1)
                    return self.embedding(token_id).unsqueeze(1)

                beams = tok0.view(-1, 1)   # [B*beam0, 1]
                lp = best_lp.view(-1)
                last_emb = emb_of_digit_token(0, beams[:, -1])
                cross_kv_exp = [expand_to_beam(kv, keep_k) for kv in cross_kv]
                # 扩展cross attention mask
                enc_key_mask_exp = expand_to_beam(enc_key_mask, keep_k)

                for d in range(1, self.n_digit):
                    x, present = self._decode_step(last_emb, cross_kv_exp, past_kv=beam_self_kv, enc_key_mask=enc_key_mask_exp)
                    logit = self._compute_digit_logits(x[:, 0, :], digit=d)
                    logp = F.log_softmax(logit, dim=-1)
                    logp = self._constrain_log_probs(logp, prefixes=beams, digit=d)

                    pre = pre_cut[d]
                    # 防止 keep 超过候选的上限（父数 * pre）
                    keep = min(beam_num[d], (beams.size(0) // B) * pre)
                    tk_prob, tk_idx = torch.topk(logp, k=pre, dim=-1)  # [N,pre]
                    cand_lp = (lp.unsqueeze(1) + tk_prob).view(B, -1)
                    best_lp, best_flat = torch.topk(cand_lp, k=keep, dim=-1)
                    parent = best_flat // pre
                    token = best_flat % pre

                    N = beams.size(0) // B
                    idx = (torch.arange(B, device=device).unsqueeze(1)*N + parent).view(-1)
                    # Select the child from the *chosen parent*.  The previous
                    # implementation gathered dimension 2 directly with an
                    # output-beam row, silently pairing a parent's prefix with
                    # another parent's child token from digit 1 onward.
                    candidate_tokens = tk_idx.view(B, N, pre)
                    chosen_parent_tokens = candidate_tokens.gather(
                        1, parent.unsqueeze(-1).expand(-1, -1, pre)
                    )
                    chosen_tokens = chosen_parent_tokens.gather(
                        2, token.unsqueeze(-1)
                    ).reshape(-1, 1)
                    beams = torch.cat([beams[idx], chosen_tokens], dim=1)
                    lp = best_lp.view(-1)

                    new_self_kv = []
                    for l in range(len(self.decoder_blocks)):
                        pk = beam_self_kv[l]
                        npk = present[l][0] if present[l] is not None else None
                        if pk is None or npk is None:
                            new_self_kv.append(None)
                        else:
                            old_k, old_v = pk
                            add_k, add_v = npk
                            old_k = old_k[idx]; old_v = old_v[idx]
                            add_k = add_k[idx]; add_v = add_v[idx]
                            new_self_kv.append((torch.cat([old_k, add_k], dim=1), torch.cat([old_v, add_v], dim=1)))
                    beam_self_kv = new_self_kv
                    # The released configuration kept the same beam width at
                    # every digit, hiding this dependency.  Latent decoding
                    # starts with only Z valid prefixes and then expands to a
                    # wider item beam, so cross-KV and its key mask must follow
                    # the selected parent rows just like self-KV.
                    cross_kv_exp = [kv[idx] for kv in cross_kv_exp]
                    enc_key_mask_exp = enc_key_mask_exp[idx]
                    last_emb = emb_of_digit_token(d, beams[:, -1])

                beams = beams.view(B, -1, self.n_digit)   # [B, num_beams, n_digit]
                lp = lp.view(B, -1)                       # [B, num_beams]

                # === Legality filtering among current top beams ===
                # Sort beams by log-probability descending, then take the top-K legal sequences
                K = min(TOPK, lp.size(1))
                sorted_lp, sorted_idx = torch.sort(lp, dim=-1, descending=True)
                sorted_idx_exp = sorted_idx.unsqueeze(-1).expand(-1, -1, self.n_digit)
                sorted_beams = beams.gather(1, sorted_idx_exp)  # [B, num_beams, n_digit]

                final_list = []
                num_beams = sorted_beams.size(1)
                for b in range(B):
                    selected = []
                    if not self.filter_illegal_final:
                        # Strict free-form control: return the model's raw
                        # top-K sequences.  The evaluator then exposes how
                        # many are legal catalog items via legal@K.
                        selected.extend(sorted_beams[b, :K])
                    elif (
                        self.tokenizer.sid_prefix_strategy == 'random_latent'
                        or self.tokenizer.has_item_aliases
                    ):
                        aggregation = str(
                            self.config.get(
                                'item_path_aggregation',
                                self.config.get('latent_item_aggregation', 'logsumexp'),
                            )
                        ).lower()
                        if aggregation not in ('logsumexp', 'max'):
                            raise ValueError(
                                'item_path_aggregation must be logsumexp|max, '
                                f'got {aggregation}'
                            )
                        item_scores = {}
                        representatives = {}
                        for j in range(num_beams):
                            seq = sorted_beams[b, j].tolist()
                            item_id = self.tokenizer.codebooks_to_item_id(seq)
                            if item_id is None:
                                continue
                            score = sorted_lp[b, j]
                            if item_id not in item_scores:
                                item_scores[item_id] = score
                                representatives[item_id] = sorted_beams[b, j]
                            elif aggregation == 'logsumexp':
                                item_scores[item_id] = torch.logaddexp(
                                    item_scores[item_id], score
                                )
                            else:
                                item_scores[item_id] = torch.maximum(
                                    item_scores[item_id], score
                                )
                        ranked_items = sorted(
                            item_scores,
                            key=lambda item_id: float(item_scores[item_id]),
                            reverse=True,
                        )
                        selected.extend(
                            representatives[item_id] for item_id in ranked_items[:K]
                        )
                    else:
                        # iterate over sorted candidates; keep only legal sequences
                        for j in range(num_beams):
                            seq = sorted_beams[b, j].tolist()
                            if self.tokenizer.codebooks_to_item_id(seq) is not None:
                                selected.append(sorted_beams[b, j])
                                if len(selected) >= K:
                                    break
                    # Fallbacks: ensure we always return K sequences
                    if len(selected) == 0:
                        # if no legal sequence found, fall back to the best candidate
                        selected.append(sorted_beams[b, 0])
                    while len(selected) < K:
                        selected.append(selected[-1])
                    final_list.append(torch.stack(selected, dim=0))

                final = torch.stack(final_list, dim=0)  # [B, K, n_digit]
                return final
        finally:
            if was_training:
                self.train()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # 注意：output_adapter如果是Identity()，不需要初始化
