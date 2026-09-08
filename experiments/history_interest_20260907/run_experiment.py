#!/usr/bin/env python3
"""One fixed-budget arm. Run from the immutable suite source snapshot."""
import argparse
import json
import os
from pathlib import Path
import subprocess

REPO = Path('/home/lingfengs111/codes/GR_variant/DiffGRM')
PYTHON = '/home/lingfengs111/miniconda3/envs/diffgrm/bin/python'
ARMS = {'pooled': ('pooled', 1), 'mlp': ('mlp', 1),
        'attention': ('attention', 1), 'interest2': ('attention', 2),
        'interest4': ('attention', 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--domain', choices=['science23', 'video23'], default='science23')
    parser.add_argument('--stage', choices=['smoke', 'parity', 'full', 'test', 'candidate_test'], required=True)
    parser.add_argument('--suite-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.suite_root.resolve()
    source = root / 'source'
    output = root / args.stage / args.domain / args.arm
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        print(f'Already complete: {output}', flush=True)
        return
    if (output / 'best.pt').exists():
        raise FileExistsError(f'Incomplete run exists; preserve it and use a new suite root: {output}')
    if args.stage == 'parity' and args.arm != 'pooled':
        raise ValueError('legacy parity is only defined for pooled')
    head, interests = ARMS[args.arm]
    sid_config = f'experiments/latte_comparison_pure/{args.domain}_opq4.yaml'
    ar_ckpt = REPO / f'saved/AmazonReviews2023CleanGR_{args.domain}_opq4_pure_ar_l20_v1/pytorch_model.bin'
    if not ar_ckpt.is_file():
        raise FileNotFoundError(ar_ckpt)
    command = [PYTHON, str(source / 'scripts/train_parallel_opq_drafter.py'),
        '--dataset', 'AmazonReviews2023CleanGR',
        '--common-config', str(source / 'experiments/amazon23_domains/common.yaml'),
        '--sid-config', str(source / sid_config),
        '--ar-config', str(source / 'experiments/canonical_full/ar_constrained.yaml'),
        '--diffusion-config', str(source / 'experiments/music23_transfer/guided_decoder.yaml'),
        '--ar-checkpoint', str(ar_ckpt),
        '--backbone-architecture', 'encoder_four_head', '--backbone-initialization', 'random',
        '--encoder-head-n-layer', '4', '--variant', 'pairwise',
        '--conditioner', 'diffusion_encoder', '--pair-rank', '51',
        '--history-head', head, '--n-interests', str(interests), '--interest-temperature', '1.0',
        '--epochs', '80', '--patience', '14', '--min-epochs', '16',
        '--batch-size', '256', '--eval-batch-size', '32',
        '--backbone-lr', '0.0003', '--selector-lr', '0.001',
        '--weight-decay', '0.0001', '--token-loss-weight', '0.1',
        '--proposal-k', '72', '--seed', '2026',
        '--fusion-alphas', '0,0.1,0.25,0.5,0.75,0.9,1',
        '--selection-metric', 'fused_ndcg10', '--retain-candidate-checkpoint',
        '--skip-tree-diagnostics', '--dump-selected-ranks', '--output-dir', str(output)]
    if args.stage not in ('test', 'candidate_test'):
        command += ['--validation-only']
    if args.stage == 'smoke':
        # Two batches at the real training batch size exercise peak memory,
        # checkpoint reload and AR fusion on concrete item IDs.
        command += ['--epochs', '1', '--max-train-examples', '512',
                    '--max-val-examples', '128', '--max-test-examples', '128']
    if args.stage == 'parity':
        checkpoint = REPO / f'runs/latte_comparison_pure/{args.domain}_opq4/onepass_pairwise_ar/best.pt'
        command += ['--epochs', '0', '--init-trained-checkpoint', str(checkpoint)]
    if args.stage in ('test', 'candidate_test'):
        checkpoint = root / 'full' / args.domain / args.arm / (
            'candidate_best.pt' if args.stage == 'candidate_test' else 'best.pt')
        command += ['--epochs', '0', '--init-trained-checkpoint', str(checkpoint)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    subprocess.run([PYTHON, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA required; refusing CPU fallback"'],
                   env=environment, check=True)
    (output / 'command.json').write_text(json.dumps({'argv': command, 'gpu': args.gpu}, indent=2))
    with (output / 'train.log').open('w') as log:
        subprocess.run(command, cwd=REPO, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
