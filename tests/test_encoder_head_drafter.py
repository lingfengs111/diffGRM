import torch

from genrec.models.DIFF_GRM.encoder_head_drafter import (
    EncoderOnlyFourHeadDrafter,
)


class _Tokenizer:
    sid_offset = 3
    vocab_size = 3 + 4 * 8


def test_encoder_only_four_head_shapes_and_padding():
    config = {
        'n_digit': 4,
        'codebook_size': 8,
        'n_embd': 16,
        'n_head': 4,
        'n_inner': 32,
        'encoder_head_n_layer': 2,
        'max_history_len': 5,
        'dropout': 0.0,
        'attn_pdrop': 0.0,
        'resid_pdrop': 0.0,
        'norm_type': 'layernorm',
        'norm_eps': 1e-5,
    }
    model = EncoderOnlyFourHeadDrafter(config, object(), _Tokenizer())
    history = torch.tensor(
        [
            [[1, 2, 3, 4], [2, 3, 4, 5], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [-1, -1, -1, -1]],
        ]
    )
    encoded = model({'history_sid': history}, return_loss=False)
    assert encoded.hidden_states.shape == (2, 1, 16)
    decoded = model.forward_decoder_only(
        {'encoder_hidden': encoded.hidden_states}
    )
    assert decoded.hidden_states.shape == (2, 4, 16)
    assert decoded.logits.shape == (2, 4, 8)
    assert torch.isfinite(decoded.logits).all()

