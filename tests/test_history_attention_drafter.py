import copy
from types import SimpleNamespace

import unittest
import torch
import torch.nn.functional as F

from genrec.models.DIFF_GRM.encoder_head_drafter import EncoderOnlyFourHeadDrafter
from genrec.models.DIFF_GRM.history_attention_drafter import (
    HistoryAttentionDrafter, aggregate_interest_scores, score_interest_catalog,
)
from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector, catalog_unary_scores
from scripts.train_parallel_opq_drafter import encode_history, one_pass_decode, one_pass_outputs, select_fusion, target_ranks


def config(**extra):
    return dict(n_digit=4, codebook_size=8, n_embd=16, n_head=4, n_inner=32,
                encoder_head_n_layer=1, max_history_len=5, dropout=0.,
                attn_pdrop=0., resid_pdrop=0., **extra)


TOKENIZER = SimpleNamespace(sid_offset=3, vocab_size=35)


def check_padding_invariance_and_full_gradient_path(head, interests):
    torch.manual_seed(7)
    model = HistoryAttentionDrafter(config(history_head=head, n_interests=interests), None, TOKENIZER)
    model.eval()
    short = torch.tensor([[[1, 2, 3, 4], [3, 2, 1, 0]]])
    padded = torch.cat([short, torch.full((1, 3, 4), -1)], 1)
    catalog = torch.randint(0, 8, (20, 4))
    selector = PairwisePathSelector(4, 8, 16, rank=3)
    first = one_pass_outputs(model, {'history_sid': short}, catalog, selector)[0]
    second = one_pass_outputs(model, {'history_sid': padded}, catalog, selector)[0]
    torch.testing.assert_close(first, second, atol=2e-6, rtol=1e-5)
    # An empty history must not make attention softmax produce NaNs.
    empty = one_pass_outputs(model, {'history_sid': torch.full((1, 5, 4), -1)}, catalog, selector)[0]
    assert torch.isfinite(empty).all()
    model.train()
    scores, logits, *_ = one_pass_outputs(model, {'history_sid': padded}, catalog, selector)
    loss = F.cross_entropy(scores, torch.tensor([0])) + .1 * F.cross_entropy(logits[:, 0], torch.tensor([1]))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert selector.left.grad is not None
    if interests > 1:
        assert model.interest_gate.weight.grad.abs().sum() > 0
        assert model.interest_queries.grad.abs().sum() > 0


def test_attention_reads_unpooled_history_and_masks_padding():
    model = HistoryAttentionDrafter(config(history_head='attention'), None, TOKENIZER).eval()
    history = torch.randn(1, 4, 16)
    mask = torch.tensor([[True, True, True, False]])
    original = model.forward_decoder_only({'encoder_hidden': history, 'history_mask': mask}).logits
    changed_pad = history.clone()
    changed_pad[:, 3] = 1000
    actual = model.forward_decoder_only({'encoder_hidden': changed_pad, 'history_mask': mask}).logits
    torch.testing.assert_close(original, actual)
    changed_early = history.clone()
    changed_early[:, 0] += torch.randn(16) * 5
    actual = model.forward_decoder_only({'encoder_hidden': changed_early, 'history_mask': mask}).logits
    assert not torch.allclose(original, actual, atol=1e-7)


def test_mixture_scores_complete_tuples_before_aggregation():
    # Each branch favors a different complete tuple; token-wise mixing would
    # give the hybrid (0,1) the same score as the coherent tuples.
    logits = torch.tensor([[[[8., -8.], [8., -8.]], [[-8., 8.], [-8., 8.]]]])
    catalog = torch.tensor([[0, 0], [1, 1], [0, 1]])
    decoded = SimpleNamespace(logits=logits, hidden_states=torch.zeros(1, 2, 2, 4),
                              interest_log_weights=torch.tensor([[.5, .5]]).log())
    model = SimpleNamespace(training=True, interest_temperature=1.)
    scores, token_probs, *_ = score_interest_catalog(model, decoded, catalog)
    assert scores[0, 0] > scores[0, 2] + 10
    assert scores[0, 1] > scores[0, 2] + 10
    torch.testing.assert_close(token_probs.exp().sum(-1), torch.ones(1, 2))
    expected = aggregate_interest_scores(
        catalog_unary_scores(logits.reshape(2, 2, 2), catalog).reshape(1, 2, 3),
        decoded.interest_log_weights,
    )
    torch.testing.assert_close(scores, expected)


def test_mixture_permutation_and_single_branch_equivalence():
    scores = torch.randn(2, 3, 9)
    weights = torch.randn(2, 3).log_softmax(-1)
    expected = aggregate_interest_scores(scores, weights, .7)
    actual = aggregate_interest_scores(scores[:, [2, 0, 1]], weights[:, [2, 0, 1]], .7)
    torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(aggregate_interest_scores(scores[:, :1], torch.zeros(2, 1)), scores[:, 0])


def test_legacy_pooled_interface_and_strict_checkpoint_reload():
    model = EncoderOnlyFourHeadDrafter(config(), None, TOKENIZER).eval()
    restored = EncoderOnlyFourHeadDrafter(config(), None, TOKENIZER).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    batch = {'history_sid': torch.randint(0, 8, (2, 5, 4))}
    encoded = encode_history(model, batch)
    assert encoded.shape == (2, 1, 16)
    torch.testing.assert_close(one_pass_decode(model, encoded).logits,
                               one_pass_decode(restored, encode_history(restored, batch)).logits)


def test_fusion_selection_uses_final_metric_and_missing_target_rank():
    validation = {'drafter_recall@72': .9, 'fused_a0p5_ndcg@10': .03,
                  'fused_a0p75_ndcg@10': .04}
    assert select_fusion(validation, [.5, .75]) == (.75, .04)
    codes = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert target_ranks(codes, torch.tensor([[3, 4], [5, 6]])).tolist() == [2, 3]


class HistoryHeadTests(unittest.TestCase):
    def test_padding_and_gradients(self):
        for head, interests in [('mlp', 1), ('attention', 1), ('attention', 2), ('attention', 4)]:
            with self.subTest(head=head, interests=interests):
                check_padding_invariance_and_full_gradient_path(head, interests)


def load_tests(loader, tests, pattern):
    del loader, pattern
    for name, function in list(globals().items()):
        if name.startswith('test_') and callable(function):
            tests.addTest(unittest.FunctionTestCase(function))
    return tests


if __name__ == '__main__':
    unittest.main()
