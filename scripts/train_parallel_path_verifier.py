#!/usr/bin/env python
"""Distill AR path knowledge into a lightweight parallel candidate verifier.

Stage 2 freezes the semantic drafter and trains only the verifier.  Stage 3 is
the same controlled pipeline with ``--joint-drafter-weight > 0``: the shared
history backbone and proposal head additionally receive a catalog-coverage
loss while the verifier retains its candidate-ranking objective.
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
from genrec.models.DIFF_GRM.path_verifier import (
    IndependentSIDHistoryEncoder,
    ParallelPathVerifier,
)
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
    parser.add_argument('--teacher-ar-checkpoint', default=None)
    parser.add_argument(
        '--residual-over-teacher-ar', action='store_true',
        help=(
            'Treat the lightweight verifier output as a residual added to a '
            'frozen AR path score. The score head is zero-initialized so '
            'epoch 0 exactly reproduces the frozen-AR verifier.'
        ),
    )
    parser.add_argument(
        '--train-on-proposal-set', action='store_true',
        help=(
            'Train on the drafter actual top-K set instead of forcibly '
            'inserting the target. Label loss is applied only when the '
            'target was genuinely recalled.'
        ),
    )
    parser.add_argument(
        '--init-verifier-checkpoint',
        default=None,
        help='Optional best.pt from a frozen-drafter verifier run.',
    )
    parser.add_argument('--pair-rank', type=int, default=51)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--num-negatives', type=int, default=15)
    parser.add_argument('--proposal-k', type=int, default=72)
    parser.add_argument('--candidate-score-chunk-size', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--coordinate-layers', type=int, default=1)
    parser.add_argument('--set-layers', type=int, default=1)
    parser.add_argument(
        '--independent-history-layers',
        type=int,
        default=0,
        help='Use a separate causal SID history encoder instead of drafter states.',
    )
    parser.add_argument('--independent-history-hidden-dim', type=int, default=256)
    parser.add_argument('--independent-history-heads', type=int, default=4)
    parser.add_argument('--independent-history-inner-dim', type=int, default=512)
    parser.add_argument(
        '--coordinate-mode',
        choices=('bidirectional', 'causal', 'mlp'),
        default='bidirectional',
    )
    parser.add_argument(
        '--history-pooling', choices=('mean', 'last'), default='mean'
    )
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--label-weight', type=float, default=1.0)
    parser.add_argument('--distill-weight', type=float, default=0.5)
    parser.add_argument('--distill-temperature', type=float, default=1.0)
    parser.add_argument('--margin-weight', type=float, default=0.1)
    parser.add_argument('--margin-value', type=float, default=0.2)
    parser.add_argument('--joint-drafter-weight', type=float, default=0.0)
    parser.add_argument('--drafter-learning-rate', type=float, default=3e-5)
    parser.add_argument('--selector-learning-rate', type=float, default=3e-4)
    parser.add_argument('--fusion-alphas', default='0,0.25,0.5,0.75,1')
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


def load_ar_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state


def hard_candidate_batch(
    proposal_scores,
    targets,
    target_rows,
    catalog,
    num_negatives,
):
    negative_scores = proposal_scores.detach().clone()
    negative_scores.scatter_(1, target_rows[:, None], float('-inf'))
    keep = min(int(num_negatives), catalog.shape[0] - 1)
    hard_rows = torch.topk(negative_scores, k=keep, dim=1).indices
    rows = torch.cat([target_rows[:, None], hard_rows], dim=1)
    candidates = torch.cat([targets[:, None, :], catalog[hard_rows]], dim=1)
    scores = proposal_scores.gather(1, rows)
    return candidates, scores


@torch.no_grad()
def evaluate(
    drafter,
    selector,
    verifier,
    history_encoder,
    loader,
    catalog,
    proposal_k,
    fusion_alphas,
    description,
    teacher=None,
    residual_over_teacher_ar=False,
    candidate_score_chunk_size=None,
):
    drafter.eval()
    selector.eval()
    verifier.eval()
    if teacher is not None:
        teacher.eval()
    if history_encoder is not None:
        history_encoder.eval()
    aggregates = {}
    candidate_hits = 0.0
    n_examples = 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc=description):
        labels = batch['labels'].to(catalog.device)
        history_hidden = encode_history(drafter, batch)
        scores, _, _, _, _ = one_pass_outputs(
            drafter,
            batch,
            catalog,
            selector,
            encoder_hidden=history_hidden,
        )
        keep = min(int(proposal_k), catalog.shape[0])
        proposal_scores, proposal_rows = torch.topk(scores, k=keep, dim=1)
        candidates = catalog[proposal_rows]
        history_mask = batch['history_sid'].to(catalog.device).ne(-1).any(dim=-1)
        verifier_history = (
            history_encoder(batch['history_sid'].to(catalog.device))
            if history_encoder is not None else history_hidden
        )
        residual_scores = verifier(
            verifier_history,
            candidates,
            proposal_scores,
            history_mask,
        )
        teacher_scores = None
        if teacher is not None:
            teacher_scores = teacher.score_candidate_paths(
                batch,
                candidates,
                chunk_size=candidate_score_chunk_size,
            )
        verifier_scores = (
            teacher_scores + residual_scores
            if residual_over_teacher_ar else residual_scores
        )
        source_scores = {
            'drafter': proposal_scores,
            'verifier': verifier_scores,
        }
        if teacher_scores is not None:
            source_scores['teacher_ar'] = teacher_scores
        for alpha in fusion_alphas:
            tag = f'{float(alpha):g}'.replace('.', 'p')
            source_scores[f'fused_a{tag}'] = (
                (1.0 - float(alpha)) * normalize_scores(proposal_scores)
                + float(alpha) * normalize_scores(verifier_scores)
            )
        for source, source_score in source_scores.items():
            order = source_score.argsort(dim=1, descending=True)
            ranked = candidates.gather(
                1, order.unsqueeze(-1).expand_as(candidates)
            )
            metrics = ranking_metrics(ranked, labels, cutoffs=(5, 10, keep))
            for name, values in metrics.items():
                aggregates.setdefault(f'{source}_{name}', []).extend(
                    values.cpu().tolist()
                )
        candidate_hit = candidates.eq(labels[:, None, :]).all(dim=-1).any(dim=1)
        candidate_hits += float(candidate_hit.sum())
        n_examples += labels.shape[0]

    result = {name: float(np.mean(values)) for name, values in aggregates.items()}
    candidate_recall = candidate_hits / max(n_examples, 1)
    result[f'candidate_recall@{min(int(proposal_k), catalog.shape[0])}'] = (
        candidate_recall
    )
    conditional_sources = ['verifier']
    if teacher is not None:
        conditional_sources.append('teacher_ar')
    conditional_sources.extend(
        f'fused_a{f"{float(alpha):g}".replace(".", "p")}'
        for alpha in fusion_alphas
    )
    for source in conditional_sources:
        result[f'{source}_conditional_recall@10'] = (
            result[f'{source}_recall@10'] / candidate_recall
            if candidate_recall else 0.0
        )
    elapsed = time.perf_counter() - started
    result.update(
        n_examples=n_examples,
        elapsed_seconds=elapsed,
        milliseconds_per_example=1000.0 * elapsed / max(n_examples, 1),
    )
    return result


def main():
    args = parse_args()
    if args.num_negatives < 1 or args.proposal_k < 1:
        raise ValueError('negative and proposal counts must be positive')
    if args.distill_temperature <= 0:
        raise ValueError('distill temperature must be positive')
    if (
        args.distill_weight or args.residual_over_teacher_ar
    ) and not args.teacher_ar_checkpoint:
        raise ValueError(
            '--teacher-ar-checkpoint is required for AR distillation/residuals'
        )
    if args.residual_over_teacher_ar and not args.train_on_proposal_set:
        raise ValueError(
            '--residual-over-teacher-ar requires --train-on-proposal-set'
        )
    if args.residual_over_teacher_ar and args.joint_drafter_weight:
        raise ValueError(
            'the guarded residual control freezes the drafter by design'
        )
    for name in (
        'label_weight', 'distill_weight', 'margin_weight',
        'joint_drafter_weight',
    ):
        if getattr(args, name) < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be non-negative')
    fusion_alphas = [float(value) for value in args.fusion_alphas.split(',')]
    if any(alpha < 0 or alpha > 1 for alpha in fusion_alphas):
        raise ValueError('fusion alphas must lie in [0,1]')

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
        raise ValueError('parallel verification requires collision-free SIDs')
    catalog = torch.as_tensor(raw_catalog, dtype=torch.long, device=device)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        tokenized['train'], batch_size=args.batch_size, shuffle=True,
        generator=train_generator,
        collate_fn=tokenizer.collate_fn['train'],
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
    drafter.load_state_dict(
        torch.load(args.diffusion_checkpoint, map_location=device)
    )
    selector = PairwisePathSelector(
        drafter.n_digit,
        drafter.codebook_size,
        drafter.n_embd,
        rank=args.pair_rank,
    ).to(device)
    drafter_state = torch.load(args.drafter_checkpoint, map_location=device)
    drafter.load_state_dict(drafter_state['model'])
    selector.load_state_dict(drafter_state['selector'])

    teacher = None
    if args.distill_weight or args.residual_over_teacher_ar:
        teacher = AR_GRM(ar_config, dataset, tokenizer).to(device)
        teacher.load_state_dict(load_ar_state(args.teacher_ar_checkpoint, device))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    verifier = ParallelPathVerifier(
        drafter.n_digit,
        drafter.codebook_size,
        (
            args.independent_history_hidden_dim
            if args.independent_history_layers else drafter.n_embd
        ),
        hidden_dim=args.hidden_dim,
        n_head=args.n_head,
        coordinate_layers=args.coordinate_layers,
        set_layers=args.set_layers,
        dropout=args.dropout,
        coordinate_mode=args.coordinate_mode,
        history_pooling=args.history_pooling,
    ).to(device)
    if args.residual_over_teacher_ar and not args.init_verifier_checkpoint:
        # A zero residual makes the untrained model exactly equal to the
        # frozen AR scorer. Validation can therefore retain epoch 0 whenever
        # candidate-aware adaptation is harmful.
        torch.nn.init.zeros_(verifier.score.weight)
        if verifier.score.bias is not None:
            torch.nn.init.zeros_(verifier.score.bias)
    history_encoder = None
    if args.independent_history_layers:
        history_encoder = IndependentSIDHistoryEncoder(
            drafter.n_digit,
            drafter.codebook_size,
            args.independent_history_hidden_dim,
            args.independent_history_heads,
            args.independent_history_layers,
            diffusion_config['max_history_len'],
            dropout=args.dropout,
            inner_dim=args.independent_history_inner_dim,
        ).to(device)
    if args.init_verifier_checkpoint:
        initialized = torch.load(
            args.init_verifier_checkpoint, map_location=device
        )
        verifier.load_state_dict(
            initialized.get('verifier', initialized)
        )
        if history_encoder is not None and initialized.get('history_encoder'):
            history_encoder.load_state_dict(initialized['history_encoder'])
    joint = args.joint_drafter_weight > 0
    for parameter in drafter.parameters():
        parameter.requires_grad_(joint)
    for parameter in selector.parameters():
        parameter.requires_grad_(joint)
    parameter_groups = [{
        'params': verifier.parameters(),
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
    }]
    if history_encoder is not None:
        parameter_groups.append({
            'params': history_encoder.parameters(),
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
        })
    if joint:
        parameter_groups.extend([
            {
                'params': drafter.parameters(),
                'lr': args.drafter_learning_rate,
                'weight_decay': args.weight_decay,
            },
            {
                'params': selector.parameters(),
                'lr': args.selector_learning_rate,
                'weight_decay': args.weight_decay,
            },
        ])
    optimizer = torch.optim.AdamW(parameter_groups)

    history = []
    best_score = float('-inf')
    best_epoch = 0
    no_improve = 0
    checkpoint_path = output_dir / 'best.pt'
    if args.init_verifier_checkpoint or args.residual_over_teacher_ar:
        initial_validation = evaluate(
            drafter,
            selector,
            verifier,
            history_encoder,
            val_loader,
            catalog,
            args.proposal_k,
            fusion_alphas,
            'parallel verifier guarded initial validation',
            teacher=teacher if args.residual_over_teacher_ar else None,
            residual_over_teacher_ar=args.residual_over_teacher_ar,
            candidate_score_chunk_size=args.candidate_score_chunk_size,
        )
        initial_alpha_scores = {
            alpha: initial_validation[
                f'fused_a{f"{float(alpha):g}".replace(".", "p")}_ndcg@10'
            ]
            for alpha in fusion_alphas
        }
        initial_alpha = max(initial_alpha_scores, key=initial_alpha_scores.get)
        best_score = initial_alpha_scores[initial_alpha]
        history.append({
            'epoch': 0,
            'selected_fusion_alpha': initial_alpha,
            'selection_ndcg@10': best_score,
            'validation': initial_validation,
        })
        # Preserve the resumed model as a valid best point.  A worse first
        # continuation epoch must not overwrite the original checkpoint.
        torch.save(
            {
                'verifier': verifier.state_dict(),
                'history_encoder': (
                    history_encoder.state_dict()
                    if history_encoder is not None else None
                ),
                'drafter': drafter.state_dict() if joint else None,
                'selector': selector.state_dict() if joint else None,
                'selected_fusion_alpha': initial_alpha,
                'validation': initial_validation,
                'epoch': 0,
                'args': vars(args),
            },
            checkpoint_path,
        )
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        verifier.train()
        if history_encoder is not None:
            history_encoder.train()
        if joint:
            drafter.train()
            selector.train()
        else:
            drafter.eval()
            selector.eval()
        metric_names = [
            'loss', 'label_loss', 'distill_loss', 'margin_loss',
            'drafter_loss', 'student_top1', 'positive_rate',
        ]
        if teacher is not None:
            metric_names.append('teacher_top1')
        epoch_values = {name: [] for name in metric_names}
        for batch in tqdm(train_loader, desc=f'parallel verifier epoch {epoch}'):
            optimizer.zero_grad(set_to_none=True)
            targets = batch['decoder_labels'].to(device)
            target_rows = code_rows(targets, catalog, drafter.codebook_size)
            grad_context = torch.enable_grad() if joint else torch.no_grad()
            with grad_context:
                history_hidden = encode_history(drafter, batch)
                proposal_scores, _, _, _, _ = one_pass_outputs(
                    drafter,
                    batch,
                    catalog,
                    selector,
                    encoder_hidden=history_hidden,
                )
                if args.train_on_proposal_set:
                    keep = min(int(args.proposal_k), catalog.shape[0])
                    candidate_proposal_scores, candidate_rows = torch.topk(
                        proposal_scores, k=keep, dim=1
                    )
                    candidates = catalog[candidate_rows]
                else:
                    candidates, candidate_proposal_scores = hard_candidate_batch(
                        proposal_scores,
                        targets,
                        target_rows,
                        catalog,
                        args.num_negatives,
                    )
                drafter_loss = F.cross_entropy(proposal_scores, target_rows)
            history_mask = batch['history_sid'].to(device).ne(-1).any(dim=-1)
            verifier_history = (
                history_encoder(batch['history_sid'].to(device))
                if history_encoder is not None else history_hidden
            )
            residual_scores = verifier(
                verifier_history,
                candidates,
                candidate_proposal_scores,
                history_mask,
            )
            teacher_scores = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_scores = teacher.score_candidate_paths(
                        batch,
                        candidates,
                        chunk_size=args.candidate_score_chunk_size,
                    )
            student_scores = (
                teacher_scores.detach() + residual_scores
                if args.residual_over_teacher_ar else residual_scores
            )
            if args.train_on_proposal_set:
                target_matches = candidates.eq(targets[:, None, :]).all(dim=-1)
                has_positive = target_matches.any(dim=1)
                labels = target_matches.long().argmax(dim=1)
            else:
                has_positive = torch.ones(
                    student_scores.shape[0], dtype=torch.bool, device=device
                )
                labels = torch.zeros(
                    student_scores.shape[0], dtype=torch.long, device=device
                )
            if has_positive.any():
                label_loss = F.cross_entropy(
                    student_scores[has_positive], labels[has_positive]
                )
            else:
                label_loss = student_scores.sum() * 0.0
            distill_loss = student_scores.new_zeros(())
            if teacher_scores is not None:
                temperature = float(args.distill_temperature)
                distill_loss = F.kl_div(
                    F.log_softmax(student_scores / temperature, dim=1),
                    F.softmax(teacher_scores / temperature, dim=1),
                    reduction='batchmean',
                ) * temperature ** 2
            if has_positive.any():
                positive_scores = student_scores.gather(
                    1, labels[:, None]
                ).squeeze(1)
                negative_scores = student_scores.masked_fill(
                    torch.nn.functional.one_hot(
                        labels, num_classes=student_scores.shape[1]
                    ).bool(),
                    float('-inf'),
                ).max(dim=1).values
                positive_margin = (
                    positive_scores[has_positive]
                    - negative_scores[has_positive]
                )
                margin_loss = F.relu(
                    float(args.margin_value) - positive_margin
                ).mean()
            else:
                margin_loss = student_scores.sum() * 0.0
            loss = (
                float(args.label_weight) * label_loss
                + float(args.distill_weight) * distill_loss
                + float(args.margin_weight) * margin_loss
                + float(args.joint_drafter_weight) * drafter_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            if history_encoder is not None:
                torch.nn.utils.clip_grad_norm_(history_encoder.parameters(), 1.0)
            if joint:
                torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()

            values = {
                'loss': loss,
                'label_loss': label_loss,
                'distill_loss': distill_loss,
                'margin_loss': margin_loss,
                'drafter_loss': drafter_loss,
                'student_top1': (
                    student_scores.argmax(dim=1)[has_positive]
                    .eq(labels[has_positive]).float().mean()
                    if has_positive.any() else student_scores.new_zeros(())
                ),
                'positive_rate': has_positive.float().mean(),
            }
            if teacher_scores is not None:
                values['teacher_top1'] = (
                    teacher_scores.argmax(dim=1)[has_positive]
                    .eq(labels[has_positive]).float().mean()
                    if has_positive.any() else teacher_scores.new_zeros(())
                )
            for name, value in values.items():
                epoch_values[name].append(float(value.detach()))

        validation = evaluate(
            drafter,
            selector,
            verifier,
            history_encoder,
            val_loader,
            catalog,
            args.proposal_k,
            fusion_alphas,
            f'parallel verifier validation epoch {epoch}',
            teacher=teacher if args.residual_over_teacher_ar else None,
            residual_over_teacher_ar=args.residual_over_teacher_ar,
            candidate_score_chunk_size=args.candidate_score_chunk_size,
        )
        alpha_scores = {
            alpha: validation[
                f'fused_a{f"{float(alpha):g}".replace(".", "p")}_ndcg@10'
            ]
            for alpha in fusion_alphas
        }
        selected_alpha = max(alpha_scores, key=alpha_scores.get)
        score = alpha_scores[selected_alpha]
        record = {
            'epoch': epoch,
            'selected_fusion_alpha': selected_alpha,
            'selection_ndcg@10': score,
            'train': {
                name: float(np.mean(values))
                for name, values in epoch_values.items()
            },
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
                    'verifier': verifier.state_dict(),
                    'history_encoder': (
                        history_encoder.state_dict()
                        if history_encoder is not None else None
                    ),
                    'drafter': drafter.state_dict() if joint else None,
                    'selector': selector.state_dict() if joint else None,
                    'selected_fusion_alpha': selected_alpha,
                    'validation': validation,
                    'args': vars(args),
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    best = torch.load(checkpoint_path, map_location=device)
    verifier.load_state_dict(best['verifier'])
    if history_encoder is not None:
        history_encoder.load_state_dict(best['history_encoder'])
    if best['drafter'] is not None:
        drafter.load_state_dict(best['drafter'])
        selector.load_state_dict(best['selector'])
    test = evaluate(
        drafter,
        selector,
        verifier,
        history_encoder,
        test_loader,
        catalog,
        args.proposal_k,
        [best['selected_fusion_alpha']],
        'parallel verifier test',
        teacher=teacher if args.residual_over_teacher_ar else None,
        residual_over_teacher_ar=args.residual_over_teacher_ar,
        candidate_score_chunk_size=args.candidate_score_chunk_size,
    )
    report = {
        'protocol': vars(args),
        'catalog_items': int(catalog.shape[0]),
        'verifier_parameters': sum(
            parameter.numel() for parameter in verifier.parameters()
        ),
        'independent_history_parameters': (
            sum(parameter.numel() for parameter in history_encoder.parameters())
            if history_encoder is not None else 0
        ),
        'joint_drafter_training': joint,
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
