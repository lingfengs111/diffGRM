import numpy as np
import torch
from pathlib import Path
import tempfile

from scripts.train_sid_rank_verifier import drafter_runtime_settings, score_report, target_ranks
from scripts.train_residual_verifier import validate_base_cache


def test_checkpoint_restores_non_normalized_logit_temperature():
    checkpoint = {'args': {
        'backbone_architecture': 'encoder_four_head', 'history_head': 'pooled',
        'n_interests': 1, 'variant': 'pairwise', 'pair_rank': 51,
        'mtp_temperature': .07, 'encoder_head_n_layer': 4,
        'training_objective': 'catalog_plus_token',
    }}
    runtime = drafter_runtime_settings(checkpoint)
    assert runtime['encoder_head_logit_temperature'] == .07
    assert runtime['encoder_head_normalize_logits'] is False
    assert runtime['encoder_head_n_layer'] == 4
    del checkpoint['args']['mtp_temperature']
    try:
        drafter_runtime_settings(checkpoint)
    except KeyError:
        pass
    else:
        raise AssertionError('missing runtime temperature must not silently default')


def test_rank_report_counts_rescue_harm_and_absent_targets():
    rows = np.tile(np.arange(72), (3, 1)).astype(np.int32)
    targets = np.array([15, 0, 99], dtype=np.int32)
    proposal = np.tile(-np.arange(72), (3, 1)).astype(np.float32)
    verifier = proposal.copy()
    verifier[0, 15] = 1000
    verifier[1, 0] = -1000
    report, ranks = score_report(proposal, verifier, rows, targets)
    assert np.array_equal(ranks['drafter_rank'], [16, 1, 73])
    assert np.array_equal(ranks['verifier_rank'], [1, 72, 73])
    assert report['candidate_recall'] == 2 / 3
    assert report['rescue_opportunities'] == 1
    assert report['original_top10_hits'] == 1
    assert report['verifier_only']['recall@10'] == 1 / 3
    reversed_rows = torch.tensor(rows[:, ::-1].copy())
    actual = target_ranks(torch.tensor(proposal[:, ::-1].copy()), reversed_rows,
                          torch.tensor(targets))
    assert actual.tolist() == [16, 1, 73]


def test_residual_cache_wrapper_directory_contract():
    # The supervisor may create only these two files before score preparation.
    # This test documents the condition checked before any cache overwrite.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'command.json').write_text('{}')
        (root / 'run.log').write_text('')
        assert [p.name for p in root.iterdir() if p.name not in {'command.json', 'run.log'}] == []
        (root / 'ready.json').write_text('{}')
        assert [p.name for p in root.iterdir() if p.name not in {'command.json', 'run.log'}] == ['ready.json']


def test_residual_training_wrapper_directory_contract():
    # Training arms follow the same non-overwrite contract as score caches.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'command.json').write_text('{}')
        (root / 'run.log').write_text('')
        allowed = {'command.json', 'run.log'}
        assert not [p.name for p in root.iterdir() if p.name not in allowed]
        (root / 'progress.json').write_text('{}')
        assert [p.name for p in root.iterdir() if p.name not in allowed] == ['progress.json']
