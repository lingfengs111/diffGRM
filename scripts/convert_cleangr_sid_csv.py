#!/usr/bin/env python
"""Convert a CleanGR semantic_ids.csv export to DiffGRM's JSON mapping."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--digits', type=int, default=4)
    parser.add_argument('--codebook-size', type=int, default=256)
    args = parser.parse_args()

    mapping = {}
    columns = [f'sid_{digit}' for digit in range(args.digits)]
    with open(args.input, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            item = row['item_id']
            codes = [int(row[column]) for column in columns]
            if any(code < 0 or code >= args.codebook_size for code in codes):
                raise ValueError(f'out-of-range code for {item}: {codes}')
            if item in mapping:
                raise ValueError(f'duplicate item row: {item}')
            mapping[item] = codes

    code_rows = [tuple(codes) for codes in mapping.values()]
    if len(set(code_rows)) != len(code_rows):
        raise ValueError(
            f'input is not collision free: {len(code_rows) - len(set(code_rows))} '
            'duplicate code rows'
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        json.dump(mapping, handle, sort_keys=True)
        handle.write('\n')
    print(json.dumps({
        'items': len(mapping),
        'digits': args.digits,
        'unique_codes': len(set(code_rows)),
        'output': str(output),
    }))


if __name__ == '__main__':
    main()

