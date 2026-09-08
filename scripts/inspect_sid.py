#!/usr/bin/env python
"""Build/load a SID catalog and print its collision/information diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from accelerate import Accelerator

from genrec.diagnostics import catalog_codes, catalog_diagnostics
from genrec.utils import get_config, get_dataset, get_tokenizer, init_device, parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DIFF_GRM')
    parser.add_argument('--dataset', default='AmazonReviews2014')
    parser.add_argument('--config', action='append', default=None)
    return parser.parse_known_args()


def main():
    args, unknown = parse_args()
    config = get_config(
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config,
        config_dict=parse_command_line_args(unknown),
    )
    config['device'], config['use_ddp'] = init_device()
    config['accelerator'] = Accelerator()
    dataset = get_dataset(args.dataset)(config)
    dataset.split()
    tokenizer = get_tokenizer(args.model)(config, dataset)
    codes = catalog_codes(tokenizer, config['codebook_size'])
    report = catalog_diagnostics(tokenizer, config['codebook_size'])
    report['codes_sha256'] = hashlib.sha256(codes.tobytes()).hexdigest()
    report['sid_quantizer'] = config['sid_quantizer']
    report['sid_collision_strategy'] = config.get('sid_collision_strategy', 'none')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
