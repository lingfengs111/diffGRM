import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch


# The lightweight test environment omits Hugging Face ``datasets``.  The AR
# model only needs its type definitions here, so provide the minimal import
# surface before loading the model module directly.
if importlib.util.find_spec('datasets') is None:
    datasets_stub = ModuleType('datasets')
    datasets_stub.Dataset = object
    sys.modules['datasets'] = datasets_stub

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'genrec/models/AR_GRM/model.py'
)
SPEC = importlib.util.spec_from_file_location('ar_model_under_test', MODULE_PATH)
AR_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AR_MODULE)
AR_GRM = AR_MODULE.AR_GRM


def tiny_model():
    config = {
        'n_digit': 4,
        'codebook_size': 8,
        'n_embd': 32,
        'n_head': 4,
        'n_inner': 64,
        'dropout': 0.0,
        'attn_pdrop': 0.0,
        'resid_pdrop': 0.0,
        'encoder_n_layer': 2,
        'decoder_n_layer': 2,
        'max_history_len': 5,
        'use_causal_mask': True,
        'share_decoder_output_embedding': True,
        'constrained_beam': False,
    }
    tokenizer = SimpleNamespace(vocab_size=34, sid_offset=2)
    return AR_GRM(config, None, tokenizer).eval()


def test_cached_candidate_scores_match_reference_and_encode_once():
    torch.manual_seed(7)
    model = tiny_model()
    history = torch.randint(0, 8, (3, 5, 4))
    history[0, :2] = -1
    candidates = torch.randint(0, 8, (3, 19, 4))
    batch = {'history_sid': history}

    encoder_calls = 0

    def count_encoder_calls(_module, _inputs, _output):
        nonlocal encoder_calls
        encoder_calls += 1

    handle = model.encoder_blocks[0].register_forward_hook(count_encoder_calls)
    with torch.no_grad():
        reference = model.score_candidate_paths(
            batch, candidates, chunk_size=6, use_cached_history=False
        )
        uncached_encoder_calls = encoder_calls
        encoder_calls = 0
        cached = model.score_candidate_paths(
            batch, candidates, chunk_size=6, use_cached_history=True
        )
        cached_encoder_calls = encoder_calls
    handle.remove()

    torch.testing.assert_close(reference, cached, atol=1e-6, rtol=1e-6)
    assert torch.equal(
        reference.argsort(dim=1, descending=True),
        cached.argsort(dim=1, descending=True),
    )
    assert uncached_encoder_calls == 4
    assert cached_encoder_calls == 1


def test_cached_candidate_training_reaches_cross_attention_kv_projection():
    torch.manual_seed(11)
    model = tiny_model().train()
    history = torch.randint(0, 8, (2, 5, 4))
    candidates = torch.randint(0, 8, (2, 5, 4))
    scores = model.score_candidate_paths(
        {'history_sid': history},
        candidates,
        chunk_size=2,
        use_cached_history=True,
    )
    torch.nn.functional.cross_entropy(
        scores, torch.zeros(2, dtype=torch.long)
    ).backward()

    gradient = model.decoder_blocks[-1].cross_attn.qkv.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_external_encoded_history_matches_cached_public_scorer():
    torch.manual_seed(17)
    model = tiny_model()
    history = torch.randint(0, 8, (2, 5, 4))
    history[0, :2] = -1
    candidates = torch.randint(0, 8, (2, 11, 4))
    batch = {'history_sid': history}

    with torch.no_grad():
        expected = model.score_candidate_paths(
            batch, candidates, chunk_size=4
        )
        encoded = model.forward(batch, return_loss=False).hidden_states
        actual = model.score_candidate_paths_from_encoded_history(
            encoded,
            history.ne(-1).any(dim=-1),
            candidates,
            chunk_size=4,
        )

    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)


def test_external_encoded_history_keeps_decoder_and_context_gradients():
    torch.manual_seed(23)
    model = tiny_model().train()
    encoded = torch.randn(2, 5, model.n_embd, requires_grad=True)
    history_mask = torch.ones(2, 5, dtype=torch.bool)
    candidates = torch.randint(0, 8, (2, 3, 4))
    logits = model.candidate_logits_from_encoded_history(
        encoded, history_mask, candidates, chunk_size=2
    )
    labels = candidates[:, 0]
    loss = sum(
        torch.nn.functional.cross_entropy(logits[:, 0, digit], labels[:, digit])
        for digit in range(model.n_digit)
    ) / model.n_digit
    loss.backward()

    assert encoded.grad is not None and encoded.grad.abs().sum() > 0
    decoder_grad = model.decoder_blocks[-1].cross_attn.qkv.weight.grad
    assert decoder_grad is not None and decoder_grad.abs().sum() > 0


def test_random_latent_candidate_score_is_max_over_every_route():
    torch.manual_seed(29)
    model = tiny_model()
    model.tokenizer.sid_prefix_strategy = 'random_latent'
    model.config['n_latent_tokens'] = 3
    model.config['latent_item_aggregation'] = 'max'
    history = torch.randint(0, 8, (2, 5, 4))
    candidates = torch.randint(0, 8, (2, 7, 4))
    batch = {'history_sid': history}

    expanded = candidates.unsqueeze(2).expand(2, 7, 3, 4).clone()
    expanded[:, :, :, 0] = torch.arange(3).view(1, 1, 3)
    expanded = expanded.reshape(2, 21, 4)
    with torch.no_grad():
        encoded = model.forward(batch, return_loss=False).hidden_states
        route_scores = model.score_candidate_paths_from_encoded_history(
            encoded,
            history.ne(-1).any(dim=-1),
            expanded,
            chunk_size=5,
        ).reshape(2, 7, 3)
        actual = model.score_candidate_paths(batch, candidates, chunk_size=5)

    torch.testing.assert_close(actual, route_scores.max(dim=-1).values)
