#!/usr/bin/env python
"""Adapt an AR path verifier to the frozen one-pass drafter history encoder.

The proposal model encodes history exactly once.  Its states feed both the
one-pass catalog head and the AR decoder, which is initialized from the
standalone AR checkpoint and adapted with the same teacher-forced token CE.
The AR item/history encoder is never called and is excluded from active-system
parameter counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector
from genrec.utils import get_config, get_dataset
from scripts.train_parallel_opq_drafter import (
    encode_history,
    normalize_scores,
    one_pass_outputs,
    ranking_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--sid-config', default=None)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument(
        '--init-shared-checkpoint',
        default=None,
        help=(
            'Optional best.pt from an earlier shared-encoder adaptation run. '
            'This resumes the active AR decoder weights while retaining the '
            'same frozen drafter/history encoder.'
        ),
    )
    parser.add_argument('--pair-rank', type=int, default=51)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument('--candidate-score-chunk-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--fusion-alphas', default='0,0.1,0.25,0.5,0.75,0.9,1')
    parser.add_argument('--max-train-examples', type=int, default=None)
    parser.add_argument('--max-val-examples', type=int, default=None)
    parser.add_argument('--max-test-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def make_config(model, dataset, files, accelerator, overrides=None):
    config = get_config(model, dataset, files, overrides or {})
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator
    return config


def load_state(path, device):
    state = torch.load(path, map_location=device)
    return state['model'] if isinstance(state, dict) and 'model' in state else state


def limit_dataset(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    return dataset.select(range(int(maximum)))


def active_ar_parameters(model):
    """Parameters used when external history states bypass the AR encoder."""
    unused_prefixes = ('item_mlp.', 'pos_emb_enc.', 'encoder_blocks.')
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(unused_prefixes)
    }


@torch.no_grad()
def evaluate(
    drafter,
    selector,
    ar_model,
    loader,
    catalog,
    proposal_k,
    fusion_alphas,
    chunk_size,
    description,
):
    drafter.eval()
    selector.eval()
    ar_model.eval()
    aggregates = {}
    candidate_hits = 0.0
    n_examples = 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc=description):
        labels = batch['labels'].to(catalog.device)
        history_hidden = encode_history(drafter, batch)
        proposal, _, _, _, _ = one_pass_outputs(
            drafter,
            batch,
            catalog,
            selector,
            encoder_hidden=history_hidden,
        )
        keep = min(int(proposal_k), catalog.shape[0])
        proposal_scores, proposal_rows = torch.topk(proposal, k=keep, dim=1)
        candidates = catalog[proposal_rows]
        history_mask = batch['history_sid'].to(catalog.device).ne(-1).any(dim=-1)
        verifier_scores = ar_model.score_candidate_paths_from_encoded_history(
            history_hidden,
            history_mask,
            candidates,
            chunk_size=chunk_size,
        )
        sources = {'drafter': proposal_scores, 'shared_ar': verifier_scores}
        for alpha in fusion_alphas:
            tag = f'{float(alpha):g}'.replace('.', 'p')
            sources[f'fused_a{tag}'] = (
                (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                + float(alpha) * normalize_scores(verifier_scores)
            )
        for source, scores in sources.items():
            order = scores.argsort(dim=1, descending=True)
            ranked = candidates.gather(1, order.unsqueeze(-1).expand_as(candidates))
            for name, values in ranking_metrics(
                ranked, labels, cutoffs=(5, 10, keep)
            ).items():
                aggregates.setdefault(f'{source}_{name}', []).extend(
                    values.cpu().tolist()
                )
        candidate_hits += float(
            candidates.eq(labels[:, None, :]).all(dim=-1).any(dim=1).sum()
        )
        n_examples += labels.shape[0]

    result = {name: float(np.mean(values)) for name, values in aggregates.items()}
    result[f'candidate_recall@{min(int(proposal_k), catalog.shape[0])}'] = (
        candidate_hits / max(n_examples, 1)
    )
    elapsed = time.perf_counter() - started
    result.update(
        n_examples=n_examples,
        elapsed_seconds=elapsed,
        milliseconds_per_example=1000.0 * elapsed / max(n_examples, 1),
    )
    return result


def selected_alpha(metrics, alphas):
    return max(
        alphas,
        key=lambda alpha: metrics[
            f'fused_a{f"{float(alpha):g}".replace(".", "p")}_ndcg@10'
        ],
    )


def main():
    args = parse_args()
    alphas = [float(value) for value in args.fusion_alphas.split(',')]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()
    files = [args.common_config] + ([args.sid_config] if args.sid_config else [])
    diffusion_config = make_config(
        'DIFF_GRM', args.dataset, files + [args.diffusion_config], accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    ar_config = make_config(
        'AR_GRM', args.dataset, files + [args.ar_config], accelerator,
        {
            'eval_batch_size': args.eval_batch_size,
            'candidate_score_chunk_size': args.candidate_score_chunk_size,
        },
    )
    device = torch.device(diffusion_config['device'])
    dataset = get_dataset(args.dataset)(diffusion_config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    raw_splits = dataset.split()
    for split, maximum in (
        ('train', args.max_train_examples),
        ('val', args.max_val_examples),
        ('test', args.max_test_examples),
    ):
        raw_splits[split] = limit_dataset(raw_splits[split], maximum)
    tokenized = tokenizer.tokenize(raw_splits)
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('shared-encoder verification requires collision-free SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        tokenized['train'], batch_size=args.batch_size, shuffle=True,
        generator=generator, collate_fn=tokenizer.collate_fn['train'],
    )
    val_loader = DataLoader(
        tokenized['val'], batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        tokenized['test'], batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )

    drafter = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    drafter.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))
    selector = PairwisePathSelector(
        drafter.n_digit, drafter.codebook_size, drafter.n_embd,
        rank=args.pair_rank,
    ).to(device)
    drafter_state = torch.load(args.drafter_checkpoint, map_location=device)
    drafter.load_state_dict(drafter_state['model'])
    selector.load_state_dict(drafter_state['selector'])
    for parameter in list(drafter.parameters()) + list(selector.parameters()):
        parameter.requires_grad_(False)

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(load_state(args.ar_checkpoint, device))
    if args.init_shared_checkpoint:
        resumed = torch.load(args.init_shared_checkpoint, map_location=device)
        if not isinstance(resumed, dict) or 'ar_model' not in resumed:
            raise ValueError(
                '--init-shared-checkpoint must contain an ar_model state dict'
            )
        ar_model.load_state_dict(resumed['ar_model'])
        print(
            f'[INIT] resumed shared AR weights from '
            f'{args.init_shared_checkpoint}',
            flush=True,
        )
    active = active_ar_parameters(ar_model)
    for name, parameter in ar_model.named_parameters():
        parameter.requires_grad_(name in active)
    optimizer = torch.optim.AdamW(
        list(active.values()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    initial_validation = evaluate(
        drafter, selector, ar_model, val_loader, catalog, args.proposal_k,
        alphas, args.candidate_score_chunk_size, 'shared AR initial validation',
    )
    initial_alpha = selected_alpha(initial_validation, alphas)
    initial_test = evaluate(
        drafter, selector, ar_model, test_loader, catalog, args.proposal_k,
        [initial_alpha], args.candidate_score_chunk_size, 'shared AR initial test',
    )

    initial_tag = f'{float(initial_alpha):g}'.replace('.', 'p')
    best_score = initial_validation[f'fused_a{initial_tag}_ndcg@10']
    best_epoch = 0
    no_improve = 0
    history = []
    checkpoint_path = output_dir / 'best.pt'
    torch.save(
        {
            'ar_model': ar_model.state_dict(),
            'selected_fusion_alpha': initial_alpha,
            'validation': initial_validation,
            'epoch': 0,
            'args': vars(args),
        },
        checkpoint_path,
    )
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        drafter.eval()
        selector.eval()
        ar_model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'shared AR epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                history_hidden = encode_history(drafter, batch)
            history_mask = batch['history_sid'].to(device).ne(-1).any(dim=-1)
            targets = batch['decoder_labels'].to(device).long()
            logits = ar_model.candidate_logits_from_encoded_history(
                history_hidden,
                history_mask,
                targets[:, None, :],
                chunk_size=1,
            )[:, 0]
            loss = sum(
                F.cross_entropy(
                    logits[:, digit],
                    targets[:, digit],
                    label_smoothing=float(ar_config.get('label_smoothing', 0.0)),
                )
                for digit in range(ar_model.n_digit)
            ) / ar_model.n_digit
            loss.backward()
            torch.nn.utils.clip_grad_norm_(active.values(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = evaluate(
            drafter, selector, ar_model, val_loader, catalog, args.proposal_k,
            alphas, args.candidate_score_chunk_size,
            f'shared AR validation epoch {epoch}',
        )
        alpha = selected_alpha(validation, alphas)
        tag = f'{float(alpha):g}'.replace('.', 'p')
        score = validation[f'fused_a{tag}_ndcg@10']
        record = {
            'epoch': epoch,
            'train_token_ce': float(np.mean(losses)),
            'selected_fusion_alpha': alpha,
            'selection_ndcg@10': score,
            'validation': validation,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    'ar_model': ar_model.state_dict(),
                    'selected_fusion_alpha': alpha,
                    'validation': validation,
                    'epoch': epoch,
                    'args': vars(args),
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    best = torch.load(checkpoint_path, map_location=device)
    ar_model.load_state_dict(best['ar_model'])
    test = evaluate(
        drafter, selector, ar_model, test_loader, catalog, args.proposal_k,
        [best['selected_fusion_alpha']], args.candidate_score_chunk_size,
        'shared AR adapted test',
    )
    backbone_parameters = sum(parameter.numel() for parameter in drafter.parameters())
    selector_parameters = sum(parameter.numel() for parameter in selector.parameters())
    active_ar_count = sum(parameter.numel() for parameter in active.values())
    report = {
        'protocol': vars(args),
        'catalog_items': int(catalog.shape[0]),
        'backbone_parameters': backbone_parameters,
        'selector_parameters': selector_parameters,
        'ar_total_parameters': sum(p.numel() for p in ar_model.parameters()),
        'ar_active_decoder_parameters': active_ar_count,
        'ar_bypassed_history_parameters': (
            sum(p.numel() for p in ar_model.parameters()) - active_ar_count
        ),
        'active_system_parameters': (
            backbone_parameters + selector_parameters + active_ar_count
        ),
        'initial_selected_fusion_alpha': initial_alpha,
        'initial_validation': initial_validation,
        'initial_test': initial_test,
        'best_epoch': best_epoch,
        'best_validation_ndcg@10': best_score,
        'selected_fusion_alpha': best['selected_fusion_alpha'],
        'history': history,
        'test': test,
        'elapsed_seconds': time.perf_counter() - started,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(test, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
