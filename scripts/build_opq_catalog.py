#!/usr/bin/env python
"""Build an OPQ catalog and emit representation-only diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from accelerate import Accelerator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.diagnostics import catalog_codes, catalog_subset_diagnostics
from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer
from genrec.utils import get_config, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2014CleanGR')
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    config = get_config('DIFF_GRM', args.dataset, args.config, {})
    config['accelerator'] = accelerator
    config['device'] = str(accelerator.device)
    config['use_ddp'] = False
    dataset = get_dataset(args.dataset)(config)
    tokenizer = DIFF_GRMTokenizer(config, dataset)
    codes = catalog_codes(tokenizer, config['codebook_size'])
    report = {
        'protocol': {
            'dataset': args.dataset,
            'configs': args.config,
            'catalog_items': int(len(codes)),
            'quantizer_tag': tokenizer._quant_tag(),
        },
        'representation': catalog_subset_diagnostics(codes),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['representation'], indent=2))
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
