import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from test_cached_candidate_verifier import tiny_model

PATH = Path(__file__).resolve().parents[1] / 'genrec/models/AR_GRM/candidate_rank_verifier.py'
SPEC = importlib.util.spec_from_file_location('candidate_rank_under_test', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CandidateRankVerifier = MODULE.CandidateRankVerifier


def inputs():
    torch.manual_seed(31)
    history = torch.randint(0, 8, (2, 5, 4))
    history[0, -2:] = -1
    candidates = torch.randint(0, 8, (2, 7, 4))
    return history, candidates


def test_causal_readout_matches_original_ar_complete_hidden():
    model = tiny_model()
    ranker = CandidateRankVerifier(model).eval()
    history, candidates = inputs()
    target = candidates[:, 0]
    with torch.no_grad():
        original = model({'history_sid': history, 'decoder_input_ids': target,
                          'decoder_labels': target})
        actual, loss = ranker(history, target[:, None], targets=target)
    torch.testing.assert_close(actual[:, 0], ranker.head(original.hidden_states[:, -1])[:, 0])
    torch.testing.assert_close(loss, original.loss)


def test_candidate_chunk_and_order_invariance():
    history, candidates = inputs()
    for attention in ('causal', 'bidirectional'):
        ranker = CandidateRankVerifier(tiny_model(), attention).eval()
        with torch.no_grad():
            full, _ = ranker(history, candidates, chunk_size=7)
            chunks, _ = ranker(history, candidates, chunk_size=2)
            order = torch.tensor([4, 2, 6, 1, 0, 5, 3])
            permuted, _ = ranker(history, candidates[:, order], chunk_size=3)
        torch.testing.assert_close(full, chunks, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(full[:, order], permuted, atol=1e-6, rtol=1e-5)


def test_only_ranking_mask_changes_and_auxiliary_remains_causal():
    history, candidates = inputs()
    causal = CandidateRankVerifier(tiny_model()).eval()
    bidir = copy.deepcopy(causal)
    bidir.attention = 'bidirectional'
    targets = candidates[:, 0]
    changed = targets.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 8
    with torch.no_grad():
        encoded = causal.encode_history(history)
        mask = history.ne(-1).any(-1)
        kv = causal.ar._precompute_cross_kv(encoded)
        ca = causal.candidate_hidden(encoded, mask, targets, kv, 1)
        cb = causal.candidate_hidden(encoded, mask, changed, kv, 1)
        ba = bidir.candidate_hidden(encoded, mask, targets, kv, 1)
        bb = bidir.candidate_hidden(encoded, mask, changed, kv, 1)
        torch.testing.assert_close(ca[:, :-1], cb[:, :-1])
        assert not torch.allclose(ba[:, 0], bb[:, 0])
        torch.testing.assert_close(causal.causal_token_loss(encoded, mask, targets),
                                   bidir.causal_token_loss(encoded, mask, targets))
        assert bidir.ar.use_causal_mask


def test_masked_history_states_cannot_affect_candidate_score():
    history, candidates = inputs()
    ranker = CandidateRankVerifier(tiny_model(), 'bidirectional').eval()
    with torch.no_grad():
        encoded = ranker.encode_history(history)
        mask = history.ne(-1).any(-1)
        changed = encoded.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 100
        torch.testing.assert_close(ranker.score_encoded(encoded, mask, candidates),
                                   ranker.score_encoded(changed, mask, candidates))


def test_rank_gradient_reaches_decoder_kv_but_not_history():
    history, candidates = inputs()
    ranker = CandidateRankVerifier(tiny_model(), 'bidirectional')
    ranker.set_decoder_trainable(True)
    ranker.train()
    scores, auxiliary = ranker(history, candidates, targets=candidates[:, 0])
    loss = F.cross_entropy(scores, torch.zeros(2, dtype=torch.long))
    loss.backward()
    for block in ranker.ar.decoder_blocks:
        grad = block.cross_attn.qkv.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad[ranker.ar.n_embd:].abs().sum() > 0
    assert ranker.head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in ranker.ar.encoder_blocks.parameters())
    assert ranker.ar.embedding.weight.grad is None
    assert ranker.ar.ln_f.weight.grad is None
    assert torch.isfinite(auxiliary)


def test_warmup_and_frozen_encoder_are_deterministic():
    history, candidates = inputs()
    ranker = CandidateRankVerifier(tiny_model())
    ranker.train()
    torch.testing.assert_close(ranker.encode_history(history), ranker.encode_history(history))
    scores, _ = ranker(history, candidates)
    F.cross_entropy(scores, torch.zeros(2, dtype=torch.long)).backward()
    assert all(p.grad is None for p in ranker.ar.parameters())
    assert ranker.head[-1].weight.grad.abs().sum() > 0


def test_zero_initialized_head_is_an_exact_residual_identity():
    history, candidates = inputs()
    for attention in ('causal', 'bidirectional'):
        ranker = CandidateRankVerifier(
            tiny_model(), attention, zero_init_head=True
        ).eval()
        with torch.no_grad():
            residual, _ = ranker(history, candidates, chunk_size=3)
        torch.testing.assert_close(residual, torch.zeros_like(residual))
