#!/usr/bin/env python
"""Distill a cached two-pass typed-set teacher into a one-pass drafter."""

import argparse
import json
from pathlib import Path
import random
import sys

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.parallel_drafter import code_rows
from genrec.utils import get_dataset
from scripts.cache_two_pass_set_teacher import selector_from_payload
from scripts.train_parallel_opq_drafter import (
    encode_history,
    evaluate,
    limit_dataset,
    make_config,
    one_pass_outputs,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return int(index), self.dataset[index]


def indexed_collate(base_collate):
    def collate(samples):
        indices, examples = zip(*samples)
        batch = base_collate(list(examples))
        batch['_distill_index'] = torch.tensor(indices, dtype=torch.long)
        return batch
    return collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--common-config', required=True)
    parser.add_argument('--sid-config', default=None)
    parser.add_argument('--ar-config', required=True)
    parser.add_argument('--diffusion-config', required=True)
    parser.add_argument('--diffusion-checkpoint', required=True)
    parser.add_argument('--student-init-checkpoint', required=True)
    parser.add_argument('--teacher-cache', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--backbone-lr', type=float, default=5e-5)
    parser.add_argument('--selector-lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--token-loss-weight', type=float, default=0.1)
    parser.add_argument('--distill-weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument('--max-train-examples', type=int, default=None)
    parser.add_argument('--max-val-examples', type=int, default=None)
    parser.add_argument('--max-test-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.temperature <= 0.0:
        raise ValueError('epochs and temperature must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()
    files = [args.common_config]
    if args.sid_config:
        files.append(args.sid_config)
    config = make_config(
        'DIFF_GRM', args.dataset,
        files + [args.diffusion_config], accelerator,
        {'train_batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size},
    )
    ar_config = make_config(
        'AR_GRM', args.dataset,
        files + [args.ar_config], accelerator,
        {'eval_batch_size': args.eval_batch_size},
    )
    device = torch.device(config['device'])
    dataset = get_dataset(args.dataset)(config)
    tokenizer = AR_GRMTokenizer(ar_config, dataset)
    raw = dataset.split()
    raw['train'] = limit_dataset(raw['train'], args.max_train_examples)
    raw['val'] = limit_dataset(raw['val'], args.max_val_examples)
    raw['test'] = limit_dataset(raw['test'], args.max_test_examples)
    tokenized = tokenizer.tokenize(raw)

    teacher = torch.load(args.teacher_cache, map_location='cpu')
    if teacher['n_examples'] < len(tokenized['train']):
        raise ValueError('teacher cache is shorter than the training dataset')
    teacher_rows = teacher['candidate_rows'][:len(tokenized['train'])]
    teacher_scores = teacher['candidate_scores'][:len(tokenized['train'])]

    train_loader = DataLoader(
        IndexedDataset(tokenized['train']),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=indexed_collate(tokenizer.collate_fn['train']),
    )
    val_loader = DataLoader(
        tokenized['val'], batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=tokenizer.collate_fn['val'],
    )
    test_loader = DataLoader(
        tokenized['test'], batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=tokenizer.collate_fn['test'],
    )
    raw_catalog = catalog_codes(tokenizer, ar_config['codebook_size'])
    if np.unique(raw_catalog, axis=0).shape[0] != raw_catalog.shape[0]:
        raise ValueError('distillation requires collision-free SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    initialized = torch.load(args.student_init_checkpoint, map_location=device)
    model = DIFF_GRM(config, dataset, tokenizer).to(device)
    model.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))
    model.load_state_dict(initialized['model'])
    selector = selector_from_payload(model, initialized, device)
    groups = [{
        'params': model.parameters(), 'lr': args.backbone_lr,
        'weight_decay': args.weight_decay,
    }]
    if selector is not None:
        groups.append({
            'params': selector.parameters(), 'lr': args.selector_lr,
            'weight_decay': args.weight_decay,
        })
    optimizer = torch.optim.AdamW(groups)

    initial = evaluate(
        model, selector, val_loader, catalog, args.proposal_k,
        description='initial one-pass validation',
    )
    initial['epoch'] = 0
    history = [initial]
    best_recall = float('-inf')
    checkpoint_path = output_dir / 'best.pt'
    temperature = float(args.temperature)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if selector is not None:
            selector.train()
        totals = {'loss': [], 'item': [], 'token': [], 'distill': []}
        for batch in tqdm(train_loader, desc=f'distill epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            indices = batch.pop('_distill_index')
            targets = batch['decoder_labels'].to(device)
            target_rows = code_rows(targets, catalog, model.codebook_size)
            encoder_hidden = encode_history(model, batch)
            scores, logits, _, _, _ = one_pass_outputs(
                model, batch, catalog, selector, encoder_hidden=encoder_hidden
            )
            item_loss = F.cross_entropy(scores, target_rows)
            token_loss = torch.stack([
                F.cross_entropy(logits[:, digit], targets[:, digit])
                for digit in range(model.n_digit)
            ]).mean()

            cached_rows = teacher_rows.index_select(0, indices).long().to(device)
            cached_scores = teacher_scores.index_select(0, indices).float().to(device)
            student_local = scores.gather(1, cached_rows)
            teacher_prob = F.softmax(cached_scores / temperature, dim=1)
            student_log_prob = F.log_softmax(student_local / temperature, dim=1)
            distill_loss = F.kl_div(
                student_log_prob, teacher_prob, reduction='batchmean'
            ) * temperature ** 2
            loss = (
                item_loss
                + float(args.token_loss_weight) * token_loss
                + float(args.distill_weight) * distill_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if selector is not None:
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            totals['loss'].append(float(loss.detach()))
            totals['item'].append(float(item_loss.detach()))
            totals['token'].append(float(token_loss.detach()))
            totals['distill'].append(float(distill_loss.detach()))

        validation = evaluate(
            model, selector, val_loader, catalog, args.proposal_k,
            description=f'one-pass validation epoch {epoch}',
        )
        validation.update(
            epoch=epoch,
            train_loss=float(np.mean(totals['loss'])),
            train_item_loss=float(np.mean(totals['item'])),
            train_token_loss=float(np.mean(totals['token'])),
            train_distill_loss=float(np.mean(totals['distill'])),
        )
        history.append(validation)
        recall = validation[f'drafter_recall@{args.proposal_k}']
        if recall > best_recall:
            best_recall = recall
            torch.save({
                'model': model.state_dict(),
                'selector': None if selector is None else selector.state_dict(),
                'args': vars(args),
                'validation': validation,
                'teacher_protocol': teacher['protocol'],
            }, checkpoint_path)

    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best['model'])
    if selector is not None:
        selector.load_state_dict(best['selector'])
    test = evaluate(
        model, selector, test_loader, catalog, args.proposal_k,
        description='distilled one-pass test',
    )
    report = {
        'protocol': vars(args),
        'teacher_protocol': teacher['protocol'],
        'best_validation': best['validation'],
        'history': history,
        'test': test,
    }
    (output_dir / 'result.json').write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(test, sort_keys=True))


if __name__ == '__main__':
    main()
