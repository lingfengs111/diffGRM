#!/usr/bin/env python
"""Matched inference-throughput benchmark for AR, diffusion, and hybrid GR.

Model/data initialization and dataloader time are excluded.  Each reported mode
uses the same examples, batch size, device, and checkpoints.  The hybrid timing
includes proposal generation, CPU union/de-duplication, AR teacher-forced path
scoring, score fusion, and ranking.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.utils import get_dataset
from scripts.evaluate_proposal_verifier import (
    _config,
    _deduplicate_candidates,
    _normalize_candidate_scores,
    _parse_orders,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2014CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--decode-orders', required=True)
    parser.add_argument('--proposal-k', type=int, default=32)
    parser.add_argument('--standalone-beam', type=int, default=128)
    parser.add_argument('--ar-output-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-examples', type=int, default=512)
    parser.add_argument('--warmup-batches', type=int, default=1)
    parser.add_argument('--fusion-alpha', type=float, default=0.5)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed_mode(name, batches, device, operation):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    candidate_counts = []
    with torch.inference_mode():
        for batch in batches:
            count = operation(batch)
            if count is not None:
                candidate_counts.append(float(count))
    synchronize(device)
    seconds = time.perf_counter() - start
    n_examples = sum(int(batch['labels'].shape[0]) for batch in batches)
    result = {
        'mode': name,
        'n_examples': n_examples,
        'seconds': seconds,
        'examples_per_second': n_examples / seconds,
        'milliseconds_per_example': 1000.0 * seconds / n_examples,
    }
    if candidate_counts:
        result['mean_candidates_per_example'] = sum(candidate_counts) / len(candidate_counts)
    if device.type == 'cuda':
        result['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
    return result


def main():
    args = parse_args()
    accelerator = Accelerator()
    common_files = [args.common_config]
    ar_config = _config(
        'AR_GRM', args.dataset, common_files + [args.ar_config], accelerator,
        {'eval_batch_size': args.batch_size},
    )
    diffusion_config = _config(
        'DIFF_GRM', args.dataset, common_files + [args.diffusion_config], accelerator,
        {
            'eval_batch_size': args.batch_size,
            'beam_search_modes': ['random'],
            'random_beam': {
                'beam_act': args.proposal_k,
                'beam_max': args.proposal_k,
                'seed': 42,
            },
            'vectorized_beam_search': {
                'top_k_final': args.proposal_k,
                'neg_inf_fp32': -1e9,
                'neg_inf_fp16': -65504.0,
                'dedup_strategy': 'simple',
                'val': {'beam_act': args.proposal_k, 'beam_max': args.proposal_k},
                'test': {'beam_act': args.proposal_k, 'beam_max': args.proposal_k},
                'beam_act': args.proposal_k,
                'beam_max': args.proposal_k,
            },
        },
    )
    device = torch.device(diffusion_config['device'])

    dataset = get_dataset(args.dataset)(ar_config)
    splits = dataset.split()
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    tokenized = tokenizer.tokenize(splits)
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

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    diffusion_model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    diffusion_model.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))
    ar_model.eval()
    diffusion_model.eval()
    diffusion_model.config['current_split'] = 'test'
    orders = _parse_orders(args.decode_orders, ar_config['n_digit'])

    def direct_ar(batch):
        ar_model.generate(batch, n_return_sequences=args.ar_output_k)

    def configure_diffusion_beam(beam_size, top_k_final):
        diffusion_model.config['random_beam'].update({
            'beam_act': beam_size,
            'beam_max': beam_size,
        })
        beam_config = diffusion_model.config['vectorized_beam_search']
        beam_config.update({
            'beam_act': beam_size,
            'beam_max': beam_size,
            'top_k_final': top_k_final,
        })
        beam_config['val'] = {'beam_act': beam_size, 'beam_max': beam_size}
        beam_config['test'] = {'beam_act': beam_size, 'beam_max': beam_size}

    def standalone_diffusion(batch):
        configure_diffusion_beam(args.standalone_beam, args.ar_output_k)
        diffusion_model.config['random_beam']['decode_order'] = orders[0]
        encoder_hidden = diffusion_model(
            batch, return_loss=False
        ).hidden_states
        diffusion_model.generate(
            batch,
            n_return_sequences=args.ar_output_k,
            mode='random',
            return_scores=False,
            encoder_hidden=encoder_hidden,
        )

    def proposals(batch, selected_orders):
        configure_diffusion_beam(args.proposal_k, args.proposal_k)
        candidate_sets = []
        score_sets = []
        encoder_hidden = diffusion_model(
            batch, return_loss=False
        ).hidden_states
        for order in selected_orders:
            diffusion_model.config['random_beam']['decode_order'] = order
            candidates, scores = diffusion_model.generate(
                batch,
                n_return_sequences=args.proposal_k,
                mode='random',
                return_scores=True,
                encoder_hidden=encoder_hidden,
            )
            candidate_sets.append(candidates)
            score_sets.append(scores)
        return _deduplicate_candidates(candidate_sets, score_sets)

    def single_diffusion(batch):
        candidates, valid, scores = proposals(batch, orders[:1])
        scores.argsort(dim=1, descending=True)
        return valid.sum(dim=1).float().mean().item()

    def multi_diffusion(batch):
        candidates, valid, scores = proposals(batch, orders)
        scores.argsort(dim=1, descending=True)
        return valid.sum(dim=1).float().mean().item()

    def hybrid(batch):
        candidates, valid, diffusion_scores = proposals(batch, orders)
        ar_scores = ar_model.score_candidate_paths(batch, candidates)
        ar_scores = ar_scores.masked_fill(~valid, float('-inf'))
        diffusion_normalized = _normalize_candidate_scores(diffusion_scores, valid)
        ar_normalized = _normalize_candidate_scores(ar_scores, valid)
        fused = (
            (1.0 - args.fusion_alpha)
            * diffusion_normalized.masked_fill(~valid, 0.0)
            + args.fusion_alpha * ar_normalized.masked_fill(~valid, 0.0)
        ).masked_fill(~valid, float('-inf'))
        fused.argsort(dim=1, descending=True)
        return valid.sum(dim=1).float().mean().item()

    warmup = batches[:max(0, args.warmup_batches)]
    if warmup:
        with torch.inference_mode():
            direct_ar(warmup[0])
            standalone_diffusion(warmup[0])
            single_diffusion(warmup[0])
            multi_diffusion(warmup[0])
            hybrid(warmup[0])
        synchronize(device)

    order_count = len(orders)
    results = [
        timed_mode('ar_constrained_beam', batches, device, direct_ar),
        timed_mode('diffusion_standalone_one_order_top10', batches, device, standalone_diffusion),
        timed_mode('diffusion_one_order_k32', batches, device, single_diffusion),
        timed_mode(f'diffusion_{order_count}_orders_k32', batches, device, multi_diffusion),
        timed_mode(
            f'diffusion_{order_count}_orders_k32_plus_ar_fusion',
            batches,
            device,
            hybrid,
        ),
    ]
    baseline = results[0]['seconds']
    for result in results:
        result['slowdown_vs_ar'] = result['seconds'] / baseline

    report = {
        'device': str(device),
        'batch_size': args.batch_size,
        'proposal_k': args.proposal_k,
        'standalone_beam': args.standalone_beam,
        'decode_orders': orders,
        'fusion_alpha': args.fusion_alpha,
        'initialization_and_dataloader_excluded': True,
        'results': results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
