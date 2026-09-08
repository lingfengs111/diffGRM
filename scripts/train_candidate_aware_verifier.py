#!/usr/bin/env python
"""Fine-tune an AR model as a proposal-aware item verifier.

The proposal model is frozen.  For every training history it retrieves hard
item negatives from the legal collision-free OPQ catalog in one parallel
pass.  The AR model is then optimized with its original token CE plus an item
listwise loss over the positive path and those proposal negatives.
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
from genrec.models.DIFF_GRM.parallel_drafter import PairwisePathSelector, code_rows
from genrec.utils import get_config, get_dataset
from scripts.train_parallel_opq_drafter import (
    evaluate,
    normalize_scores,
    one_pass_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2014CleanGR')
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--sid-config', default=None)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--pair-rank', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--token-loss-weight', type=float, default=1.0)
    parser.add_argument('--listwise-weight', type=float, required=True)
    parser.add_argument('--listwise-temperature', type=float, default=1.0)
    parser.add_argument('--margin-weight', type=float, default=0.0)
    parser.add_argument('--margin-value', type=float, default=0.2)
    parser.add_argument(
        '--training-fusion-alpha',
        type=float,
        default=0.75,
        help=(
            'AR weight used inside the residual verifier training loss. '
            'The proposal weight is 1-alpha; validation still selects from '
            'the full --fusion-alphas grid.'
        ),
    )
    parser.add_argument(
        '--trainable-scope',
        choices=('all', 'decoder', 'last_decoder'),
        default='last_decoder',
        help='Conservative verifier adaptation scope.',
    )
    parser.add_argument('--num-negatives', type=int, default=15)
    parser.add_argument('--candidate-score-chunk-size', type=int, default=4)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument(
        '--fusion-alphas', default='0,0.1,0.25,0.5,0.75,0.9,1'
    )
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


def limit_dataset(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    return dataset.select(range(int(maximum)))


def proposal_negatives(
    drafter_model,
    selector,
    batch,
    catalog,
    targets,
    num_negatives,
):
    """Return positive-first candidates and their frozen proposal scores."""
    with torch.no_grad():
        proposal_scores, _, _, _, _ = one_pass_outputs(
            drafter_model, batch, catalog, selector
        )
        target_rows = code_rows(
            targets, catalog, drafter_model.codebook_size
        )
        negative_scores = proposal_scores.clone()
        negative_scores.scatter_(1, target_rows[:, None], float('-inf'))
        keep = min(int(num_negatives), catalog.shape[0] - 1)
        hard_scores, hard_rows = torch.topk(negative_scores, k=keep, dim=1)
        hard_codes = catalog[hard_rows]
        candidates = torch.cat([targets[:, None, :], hard_codes], dim=1)
        candidate_proposal_scores = torch.cat(
            [proposal_scores.gather(1, target_rows[:, None]), hard_scores], dim=1
        )
    return candidates, candidate_proposal_scores


def validation_score(result, fusion_alphas, metric_k=10):
    scores = {}
    for alpha in fusion_alphas:
        tag = f'{float(alpha):g}'.replace('.', 'p')
        scores[float(alpha)] = result[f'fused_a{tag}_ndcg@{metric_k}']
    best_alpha = max(scores, key=scores.get)
    return float(scores[best_alpha]), float(best_alpha)


def configure_trainable_scope(model, scope):
    """Keep the strong generator intact while adapting it for verification."""
    if scope == 'all':
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        last_decoder = len(model.decoder_blocks) - 1
        for name, parameter in model.named_parameters():
            trainable = (
                name.startswith('decoder_blocks.')
                if scope == 'decoder'
                else name.startswith(f'decoder_blocks.{last_decoder}.')
            )
            if trainable or name.startswith('ln_f.') or name == 'bos_embedding':
                parameter.requires_grad_(True)
    selected = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not selected:
        raise ValueError(f'no trainable parameters for scope={scope}')
    return selected


def main():
    args = parse_args()
    if args.listwise_weight < 0:
        raise ValueError('--listwise-weight must be non-negative')
    if args.listwise_temperature <= 0:
        raise ValueError('--listwise-temperature must be positive')
    if args.margin_weight < 0 or args.margin_value < 0:
        raise ValueError('--margin-weight/value must be non-negative')
    if not 0.0 <= args.training_fusion_alpha <= 1.0:
        raise ValueError('--training-fusion-alpha must lie in [0,1]')
    if args.num_negatives < 1:
        raise ValueError('--num-negatives must be positive')
    fusion_alphas = [float(value) for value in args.fusion_alphas.split(',')]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()
    common_files = [args.common_config]
    if args.sid_config:
        common_files.append(args.sid_config)
    diffusion_config = make_config(
        'DIFF_GRM', args.dataset,
        common_files + [args.diffusion_config], accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    ar_config = make_config(
        'AR_GRM', args.dataset,
        common_files + [args.ar_config], accelerator,
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
        raise ValueError('candidate-aware verification requires collision-free SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    train_loader = DataLoader(
        limit_dataset(tokenized['train'], args.max_train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tokenizer.collate_fn['train'],
    )
    val_loader = DataLoader(
        limit_dataset(tokenized['val'], args.max_val_examples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        limit_dataset(tokenized['test'], args.max_test_examples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )

    drafter_model = DIFF_GRM(diffusion_config, dataset, tokenizer).to(device)
    drafter_model.load_state_dict(
        torch.load(args.diffusion_checkpoint, map_location=device)
    )
    selector = PairwisePathSelector(
        drafter_model.n_digit,
        drafter_model.codebook_size,
        drafter_model.n_embd,
        rank=args.pair_rank,
    ).to(device)
    drafter_state = torch.load(args.drafter_checkpoint, map_location=device)
    drafter_model.load_state_dict(drafter_state['model'])
    selector.load_state_dict(drafter_state['selector'])
    drafter_model.eval()
    selector.eval()
    for parameter in drafter_model.parameters():
        parameter.requires_grad_(False)
    for parameter in selector.parameters():
        parameter.requires_grad_(False)

    ar_model = AR_GRM(ar_config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    trainable_parameters = configure_trainable_scope(
        ar_model, args.trainable_scope
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    ar_model.eval()
    initial_validation = evaluate(
        drafter_model,
        selector,
        val_loader,
        catalog,
        args.proposal_k,
        ar_model=ar_model,
        fusion_alphas=fusion_alphas,
        description='candidate-aware initial validation',
    )
    best_score, initial_alpha = validation_score(
        initial_validation,
        fusion_alphas,
        metric_k=min(10, int(args.proposal_k), int(catalog.shape[0])),
    )
    best_epoch = 0
    no_improve = 0
    checkpoint_path = output_dir / 'best_ar.pt'
    started = time.perf_counter()
    initial_record = {
        'epoch': 0,
        'selected_fusion_alpha': initial_alpha,
        'selection_ndcg@10': best_score,
        'validation': initial_validation,
    }
    history.append(initial_record)
    torch.save(
        {
            'model': ar_model.state_dict(),
            'args': vars(args),
            'epoch': 0,
            'selected_fusion_alpha': initial_alpha,
            'validation': initial_validation,
        },
        checkpoint_path,
    )
    print(json.dumps(initial_record, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        ar_model.train()
        losses = []
        token_losses = []
        listwise_losses = []
        margin_losses = []
        positive_rates = []
        proposal_positive_rates = []
        margins = []
        for batch in tqdm(train_loader, desc=f'verifier train epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            targets = batch['decoder_labels'].to(device)
            candidates, candidate_proposal_scores = proposal_negatives(
                drafter_model,
                selector,
                batch,
                catalog,
                targets,
                args.num_negatives,
            )
            token_loss = ar_model(batch, return_loss=True).loss
            path_scores = ar_model.score_candidate_paths(
                batch,
                candidates,
                chunk_size=args.candidate_score_chunk_size,
            )
            fused_scores = (
                (1.0 - float(args.training_fusion_alpha))
                * normalize_scores(candidate_proposal_scores)
                + float(args.training_fusion_alpha)
                * normalize_scores(path_scores)
            )
            listwise_loss = F.cross_entropy(
                fused_scores / args.listwise_temperature,
                torch.zeros(fused_scores.shape[0], dtype=torch.long, device=device),
            )
            positive_margin = (
                fused_scores[:, 0]
                - fused_scores[:, 1:].max(dim=1).values
            )
            margin_loss = F.relu(
                float(args.margin_value) - positive_margin
            ).mean()
            loss = (
                float(args.token_loss_weight) * token_loss
                + float(args.listwise_weight) * listwise_loss
                + float(args.margin_weight) * margin_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach()))
            token_losses.append(float(token_loss.detach()))
            listwise_losses.append(float(listwise_loss.detach()))
            margin_losses.append(float(margin_loss.detach()))
            positive_rates.append(
                float(fused_scores.argmax(dim=1).eq(0).float().mean().detach())
            )
            proposal_positive_rates.append(
                float(
                    candidate_proposal_scores.argmax(dim=1)
                    .eq(0).float().mean().detach()
                )
            )
            margins.append(
                float(positive_margin.mean().detach())
            )

        ar_model.eval()
        validation = evaluate(
            drafter_model,
            selector,
            val_loader,
            catalog,
            args.proposal_k,
            ar_model=ar_model,
            fusion_alphas=fusion_alphas,
            description=f'candidate-aware validation epoch {epoch}',
        )
        selection_k = min(10, int(args.proposal_k), int(catalog.shape[0]))
        score, selected_alpha = validation_score(
            validation, fusion_alphas, metric_k=selection_k
        )
        epoch_record = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'train_token_loss': float(np.mean(token_losses)),
            'train_listwise_loss': float(np.mean(listwise_losses)),
            'train_margin_loss': float(np.mean(margin_losses)),
            'train_positive_top1_rate': float(np.mean(positive_rates)),
            'train_proposal_positive_top1_rate': float(
                np.mean(proposal_positive_rates)
            ),
            'train_positive_margin': float(np.mean(margins)),
            'selected_fusion_alpha': selected_alpha,
            'selection_ndcg@10': score,
            'validation': validation,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    'model': ar_model.state_dict(),
                    'args': vars(args),
                    'epoch': epoch,
                    'selected_fusion_alpha': selected_alpha,
                    'validation': validation,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    best = torch.load(checkpoint_path, map_location=device)
    ar_model.load_state_dict(best['model'])
    ar_model.eval()
    test = evaluate(
        drafter_model,
        selector,
        test_loader,
        catalog,
        args.proposal_k,
        ar_model=ar_model,
        fusion_alphas=[best['selected_fusion_alpha']],
        description='candidate-aware test',
    )
    report = {
        'protocol': vars(args),
        'catalog_items': int(catalog.shape[0]),
        'best_epoch': best_epoch,
        'best_validation_ndcg@10': best_score,
        'selected_fusion_alpha': best['selected_fusion_alpha'],
        'best_validation': best['validation'],
        'history': history,
        'test': test,
        'elapsed_seconds': time.perf_counter() - started,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(test, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
