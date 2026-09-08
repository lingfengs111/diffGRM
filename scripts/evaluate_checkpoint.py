#!/usr/bin/env python
"""Evaluate one trained checkpoint without entering the training pipeline."""

import argparse
import json
from pathlib import Path
import sys

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.utils import get_config, get_dataset, get_model, get_tokenizer, get_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='AmazonReviews2014')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', action='append', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--no-diagnostics', action='store_true')
    parser.add_argument('--item-texts-file', default=None)
    parser.add_argument('--metadata-cache-tag', default=None)
    parser.add_argument('--max-history-len', type=int, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    accelerator = Accelerator()
    overrides = {
        'decoder_diagnostics': {
            'enabled': not args.no_diagnostics,
            'splits': [args.split],
        }
    }
    if args.batch_size is not None:
        overrides['eval_batch_size'] = args.batch_size
    if args.item_texts_file is not None:
        overrides['item_texts_file'] = args.item_texts_file
    if args.metadata_cache_tag is not None:
        overrides['metadata_cache_tag'] = args.metadata_cache_tag
    if args.max_history_len is not None:
        overrides['max_history_len'] = args.max_history_len
        overrides['max_hist_len'] = args.max_history_len
    config = get_config(args.model, args.dataset, args.config, overrides)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['use_ddp'] = False
    config['accelerator'] = accelerator
    config['current_split'] = args.split

    dataset = get_dataset(args.dataset)(config)
    splits = dataset.split()
    tokenizer = get_tokenizer(args.model)(config, dataset)
    tokenized = tokenizer.tokenize(splits)
    model = get_model(args.model)(config, dataset, tokenizer)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model, loader = accelerator.prepare(
        model,
        DataLoader(
            tokenized[args.split], batch_size=config['eval_batch_size'], shuffle=False,
            collate_fn=tokenizer.collate_fn[args.split],
        ),
    )
    trainer = get_trainer(args.model)(config, model, tokenizer)
    results = trainer.evaluate(loader, split=args.split)
    rendered = json.dumps({key: float(value) for key, value in results.items()}, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
