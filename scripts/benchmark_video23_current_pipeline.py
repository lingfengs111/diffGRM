#!/usr/bin/env python
"""Matched Video23 latency benchmark for the current and earlier GR decoders.

Initialization, tokenization, dataloader construction, and host-side metric
calculation are excluded.  Every timed mode sees the same examples and batch
size.  The current pipeline timing includes one-pass pairwise proposal, cached
AR path scoring, score fusion, and ranking.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector
from genrec.utils import get_dataset
from scripts.benchmark_inference_latency import synchronize, timed_mode
from scripts.evaluate_proposal_verifier import (
    _config,
    _deduplicate_candidates,
    _normalize_candidate_scores,
    _parse_orders,
)
from scripts.train_parallel_opq_drafter import one_pass_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--pairwise-checkpoint', required=True)
    parser.add_argument(
        '--decode-orders',
        default='0,1,2,3;2,3,0,1;3,2,1,0;1,0,3,2',
    )
    parser.add_argument('--pair-rank', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-examples', type=int, default=512)
    parser.add_argument('--warmup-batches', type=int, default=1)
    parser.add_argument('--fusion-alpha', type=float, default=0.75)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    common_files = [args.common_config]
    ar_config = _config(
        'AR_GRM', args.dataset, common_files + [args.ar_config], accelerator,
        {'eval_batch_size': args.batch_size},
    )
    diffusion_config = _config(
        'DIFF_GRM', args.dataset,
        common_files + [args.diffusion_config], accelerator,
        {'eval_batch_size': args.batch_size},
    )
    device = torch.device(diffusion_config['device'])

    dataset = get_dataset(args.dataset)(ar_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    tokenized = tokenizer.tokenize(dataset.split())
    eval_data = tokenized['test'].select(
        range(min(args.max_examples, len(tokenized['test'])))
    )
    loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )
    batches = list(loader)

    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('benchmark requires collision-free catalog SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    ar_model.eval()

    original_diffusion = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    original_diffusion.load_state_dict(
        torch.load(args.diffusion_checkpoint, map_location=device)
    )
    original_diffusion.eval()
    original_diffusion.config['current_split'] = 'test'

    current_drafter = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    selector = PairwisePathSelector(
        current_drafter.n_digit,
        current_drafter.codebook_size,
        current_drafter.n_embd,
        rank=args.pair_rank,
    ).to(device)
    pairwise_state = torch.load(args.pairwise_checkpoint, map_location=device)
    current_drafter.load_state_dict(pairwise_state['model'])
    selector.load_state_dict(pairwise_state['selector'])
    current_drafter.eval()
    selector.eval()

    orders = _parse_orders(args.decode_orders, ar_config['n_digit'])

    def configure_diffusion_beam(beam_size, top_k_final):
        original_diffusion.config['random_beam'].update({
            'beam_act': beam_size,
            'beam_max': beam_size,
        })
        beam_config = original_diffusion.config['vectorized_beam_search']
        beam_config.update({
            'beam_act': beam_size,
            'beam_max': beam_size,
            'top_k_final': top_k_final,
        })
        beam_config['val'] = {'beam_act': beam_size, 'beam_max': beam_size}
        beam_config['test'] = {'beam_act': beam_size, 'beam_max': beam_size}

    def direct_ar(batch):
        ar_model.generate(batch, n_return_sequences=10)

    def original_diffusion_final(batch):
        configure_diffusion_beam(128, 10)
        original_diffusion.config['random_beam']['decode_order'] = orders[0]
        encoder_hidden = original_diffusion(
            batch, return_loss=False
        ).hidden_states
        original_diffusion.generate(
            batch,
            n_return_sequences=10,
            mode='random',
            return_scores=False,
            encoder_hidden=encoder_hidden,
        )

    def original_proposals(batch):
        configure_diffusion_beam(32, 32)
        candidate_sets = []
        score_sets = []
        encoder_hidden = original_diffusion(
            batch, return_loss=False
        ).hidden_states
        for order in orders:
            original_diffusion.config['random_beam']['decode_order'] = order
            candidates, scores = original_diffusion.generate(
                batch,
                n_return_sequences=32,
                mode='random',
                return_scores=True,
                encoder_hidden=encoder_hidden,
            )
            candidate_sets.append(candidates)
            score_sets.append(scores)
        return _deduplicate_candidates(candidate_sets, score_sets)

    def original_four_order_hybrid(batch):
        candidates, valid, diffusion_scores = original_proposals(batch)
        ar_scores = ar_model.score_candidate_paths(batch, candidates)
        ar_scores = ar_scores.masked_fill(~valid, float('-inf'))
        fused = (
            (1.0 - args.fusion_alpha)
            * _normalize_candidate_scores(diffusion_scores, valid).masked_fill(
                ~valid, 0.0
            )
            + args.fusion_alpha
            * _normalize_candidate_scores(ar_scores, valid).masked_fill(
                ~valid, 0.0
            )
        ).masked_fill(~valid, float('-inf'))
        fused.argsort(dim=1, descending=True)
        return valid.sum(dim=1).float().mean().item()

    def current_proposals(batch, candidate_k):
        scores, _, _, _, _ = one_pass_outputs(
            current_drafter,
            batch,
            catalog,
            selector,
        )
        keep = min(int(candidate_k), catalog.shape[0])
        proposal_scores, proposal_rows = torch.topk(scores, k=keep, dim=1)
        return catalog[proposal_rows], proposal_scores

    def current_drafter_only(batch, candidate_k):
        candidates, scores = current_proposals(batch, candidate_k)
        scores.argsort(dim=1, descending=True)
        return candidates.shape[1]

    def current_pipeline(batch, candidate_k):
        candidates, proposal_scores = current_proposals(batch, candidate_k)
        ar_scores = ar_model.score_candidate_paths(batch, candidates)
        fused = (
            (1.0 - args.fusion_alpha)
            * _normalize_candidate_scores(
                proposal_scores, torch.ones_like(proposal_scores, dtype=torch.bool)
            )
            + args.fusion_alpha
            * _normalize_candidate_scores(
                ar_scores, torch.ones_like(ar_scores, dtype=torch.bool)
            )
        )
        fused.argsort(dim=1, descending=True)
        return candidates.shape[1]

    operations = [
        ('standalone_ar_constrained_beam128_top10', direct_ar),
        ('original_diffgrm_one_order_beam128_top10', original_diffusion_final),
        ('original_diffgrm_four_order_k32_plus_cached_ar', original_four_order_hybrid),
        ('current_pairwise_drafter_only_k32', lambda batch: current_drafter_only(batch, 32)),
        ('current_pairwise_k32_plus_cached_ar', lambda batch: current_pipeline(batch, 32)),
        ('current_pairwise_k72_plus_cached_ar', lambda batch: current_pipeline(batch, 72)),
    ]

    warmup = batches[:max(0, args.warmup_batches)]
    if warmup:
        with torch.inference_mode():
            for _, operation in operations:
                operation(warmup[0])
        synchronize(device)

    results = [
        timed_mode(name, batches, device, operation)
        for name, operation in operations
    ]
    ar_seconds = results[0]['seconds']
    for result in results:
        result['slowdown_vs_standalone_ar'] = result['seconds'] / ar_seconds

    report = {
        'device': str(device),
        'batch_size': args.batch_size,
        'n_examples': sum(int(batch['labels'].shape[0]) for batch in batches),
        'decode_orders': orders,
        'fusion_alpha': args.fusion_alpha,
        'initialization_tokenization_dataloader_and_metrics_excluded': True,
        'results': results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
