#!/usr/bin/env python
"""Train an exact-MIPS candidate drafter and verify its top-K with AR.

Exact catalog scoring deliberately removes ANN index approximation from the
quality experiment.  The resulting learned item vectors can later be exported
to FAISS without changing the retrieval objective.
"""

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
from genrec.models.AR_GRM.ann_drafter import ExactMIPSDrafter
from genrec.models.AR_GRM.model import AR_GRM
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.parallel_drafter import code_rows
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--ar-checkpoint', required=True)
    parser.add_argument('--item-mode', choices=('id', 'opq'), required=True)
    parser.add_argument('--init-trained-checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument('--fusion-alphas', default='0,0.25,0.5,0.75,0.9,1')
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


def normalize_scores(scores):
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def ranking_metrics(ranked_codes, labels, cutoffs=(5, 10)):
    matches = ranked_codes.eq(labels[:, None, :]).all(dim=-1)
    positions = torch.arange(ranked_codes.shape[1], device=ranked_codes.device)[None]
    first = torch.where(matches, positions, ranked_codes.shape[1]).min(dim=1).values
    output = {}
    for requested in cutoffs:
        cutoff = min(int(requested), ranked_codes.shape[1])
        hits = matches[:, :cutoff].any(dim=1)
        output[f'recall@{cutoff}'] = hits.float()
        output[f'ndcg@{cutoff}'] = torch.where(
            hits,
            torch.log2(first.float() + 2.0).reciprocal(),
            torch.zeros_like(first, dtype=torch.float),
        )
    return output


@torch.no_grad()
def encode_history(ar_model, batch):
    return ar_model(batch, return_loss=False).hidden_states


@torch.no_grad()
def evaluate(
    drafter,
    ar_model,
    loader,
    catalog,
    proposal_k,
    fusion_alphas=(),
    verify=False,
    description='evaluation',
):
    drafter.eval()
    ar_model.eval()
    aggregates = {}
    n_examples = 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc=description):
        history_sid = batch['history_sid'].to(catalog.device)
        labels = batch['labels'].to(catalog.device)
        encoder_hidden = encode_history(ar_model, batch)
        scores = drafter(encoder_hidden, history_sid)
        keep = min(int(proposal_k), catalog.shape[0])
        proposal_scores, proposal_rows = torch.topk(scores, k=keep, dim=1)
        proposals = catalog[proposal_rows]
        metrics = ranking_metrics(proposals, labels, cutoffs=(5, 10, keep))
        for name, values in metrics.items():
            aggregates.setdefault(f'ann_{name}', []).extend(values.cpu().tolist())

        if verify:
            ar_scores = ar_model.score_candidate_paths(batch, proposals)
            order = ar_scores.argsort(dim=1, descending=True)
            ranked = proposals.gather(1, order.unsqueeze(-1).expand_as(proposals))
            metrics = ranking_metrics(ranked, labels)
            for name, values in metrics.items():
                aggregates.setdefault(f'ar_verified_{name}', []).extend(
                    values.cpu().tolist()
                )
            for alpha in fusion_alphas:
                fused = (
                    (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                    + float(alpha) * normalize_scores(ar_scores)
                )
                order = fused.argsort(dim=1, descending=True)
                ranked = proposals.gather(1, order.unsqueeze(-1).expand_as(proposals))
                metrics = ranking_metrics(ranked, labels)
                tag = f'{float(alpha):g}'.replace('.', 'p')
                for name, values in metrics.items():
                    aggregates.setdefault(f'fused_a{tag}_{name}', []).extend(
                        values.cpu().tolist()
                    )
        n_examples += labels.shape[0]

    elapsed = time.perf_counter() - started
    result = {name: float(np.mean(values)) for name, values in aggregates.items()}
    result.update(
        n_examples=n_examples,
        elapsed_seconds=elapsed,
        milliseconds_per_example=1000.0 * elapsed / max(n_examples, 1),
    )
    return result


def main():
    args = parse_args()
    fusion_alphas = [float(value) for value in args.fusion_alphas.split(',')]
    if not fusion_alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in fusion_alphas):
        raise ValueError('--fusion-alphas must contain values in [0,1]')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()
    config = make_config(
        'AR_GRM',
        args.dataset,
        [args.common_config, args.ar_config],
        accelerator,
        {'train_batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size},
    )
    device = torch.device(config['device'])
    dataset = get_dataset(args.dataset)(config)
    tokenizer = AR_GRMTokenizer(config, dataset)
    tokenized = tokenizer.tokenize(dataset.split())
    raw_catalog = catalog_codes(tokenizer, config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('exact-MIPS experiment requires collision-free item SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    train_data = limit_dataset(tokenized['train'], args.max_train_examples)
    val_data = limit_dataset(tokenized['val'], args.max_val_examples)
    test_data = limit_dataset(tokenized['test'], args.max_test_examples)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tokenizer.collate_fn['train'],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )

    ar_model = AR_GRM(config, dataset, tokenizer).to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    ar_model.eval()
    for parameter in ar_model.parameters():
        parameter.requires_grad_(False)

    drafter = ExactMIPSDrafter(ar_model, catalog, args.item_mode).to(device)
    if args.init_trained_checkpoint:
        initialized = torch.load(args.init_trained_checkpoint, map_location=device)
        drafter.load_state_dict(initialized['drafter'])
    if args.epochs == 0 and not args.init_trained_checkpoint:
        raise ValueError('--epochs=0 requires --init-trained-checkpoint')

    optimizer = torch.optim.AdamW(
        drafter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    best_recall = float('-inf')
    checkpoint_path = output_dir / 'best.pt'
    initial = evaluate(
        drafter, ar_model, val_loader, catalog, args.proposal_k,
        description='initial validation',
    )
    initial['epoch'] = 0
    history.append(initial)
    print(json.dumps(initial, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        drafter.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            history_sid = batch['history_sid'].to(device)
            targets = batch['decoder_labels'].to(device)
            target_rows = code_rows(targets, catalog, config['codebook_size'])
            with torch.no_grad():
                encoder_hidden = encode_history(ar_model, batch)
            scores = drafter(encoder_hidden, history_sid)
            loss = F.cross_entropy(scores, target_rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation = evaluate(
            drafter, ar_model, val_loader, catalog, args.proposal_k,
            description=f'validation epoch {epoch}',
        )
        validation.update(epoch=epoch, train_loss=float(np.mean(losses)))
        history.append(validation)
        print(json.dumps(validation, sort_keys=True), flush=True)
        selection_key = f'ann_recall@{min(args.proposal_k, catalog.shape[0])}'
        if validation[selection_key] > best_recall:
            best_recall = validation[selection_key]
            torch.save(
                {
                    'drafter': drafter.state_dict(),
                    'args': vars(args),
                    'validation': validation,
                },
                checkpoint_path,
            )

    if args.epochs == 0:
        best = {
            'drafter': drafter.state_dict(),
            'args': vars(args),
            'validation': initial,
        }
    else:
        best = torch.load(checkpoint_path, map_location=device)
    drafter.load_state_dict(best['drafter'])
    validation_with_verifier = evaluate(
        drafter,
        ar_model,
        val_loader,
        catalog,
        args.proposal_k,
        fusion_alphas=fusion_alphas,
        verify=True,
        description='validation fusion selection',
    )
    alpha_scores = {}
    for alpha in fusion_alphas:
        tag = f'{alpha:g}'.replace('.', 'p')
        alpha_scores[alpha] = validation_with_verifier[f'fused_a{tag}_ndcg@10']
    selected_alpha = max(alpha_scores, key=alpha_scores.get)
    test = evaluate(
        drafter,
        ar_model,
        test_loader,
        catalog,
        args.proposal_k,
        fusion_alphas=fusion_alphas,
        verify=True,
        description='test exact-MIPS + AR verification',
    )
    report = {
        'method': f'exact_mips_{args.item_mode}',
        'protocol': vars(args),
        'catalog_items': int(catalog.shape[0]),
        'best_validation': best['validation'],
        'validation_with_verifier': validation_with_verifier,
        'selected_fusion_alpha': selected_alpha,
        'history': history,
        'test': test,
    }
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(test, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

