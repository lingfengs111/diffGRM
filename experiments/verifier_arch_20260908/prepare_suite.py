#!/usr/bin/env python3
"""Freeze a self-contained source/checkpoint bundle; raw datasets stay external."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f'refusing to overwrite {root}')
    root.mkdir(parents=True)
    source = root / 'source'
    for directory in ('genrec', 'scripts', 'tests'):
        for path in (REPO / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.json'):
                target = source / path.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    experiment = Path('experiments/verifier_arch_20260908')
    shutil.copytree(REPO / experiment, source / experiment,
                    ignore=shutil.ignore_patterns('__pycache__'))
    provenance = {}
    for domain in ('video23', 'science23'):
        directory = root / 'inputs' / domain
        directory.mkdir(parents=True)
        config_path = REPO / f'experiments/latte_comparison_pure/{domain}_opq4.yaml'
        config = yaml.safe_load(config_path.read_text())
        sources = {
            'domain.yaml': config_path,
            'common.yaml': REPO / 'experiments/amazon23_domains/common.yaml',
            'ar.yaml': REPO / 'experiments/canonical_full/ar_constrained.yaml',
            'sid.sem_ids': Path(config['sid_override_path']),
            'ar.pt': REPO / f'saved/AmazonReviews2023CleanGR_{domain}_opq4_pure_ar_l20_v1/pytorch_model.bin',
            'drafter.pt': REPO / f'runs/history_interest_20260907/v1/full/{domain}/pooled/best.pt',
            'reference.json': REPO / f'runs/history_interest_20260907/v1/full/{domain}/pooled/result.json',
        }
        for name, origin in sources.items():
            target = directory / name
            shutil.copy2(origin, target)
            provenance[str(target.relative_to(root))] = str(origin)
    manifest = {'files': {str(p.relative_to(root)): digest(p)
                          for p in sorted(root.rglob('*')) if p.is_file()},
                'original_inputs': provenance}
    (root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(root)


if __name__ == '__main__':
    main()
