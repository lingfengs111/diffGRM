#!/usr/bin/env python3
"""Create an immutable residual-suite source snapshot and base-cache manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'runs/verifier_arch_20260908/v2/full/video23/cache'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f'refusing to overwrite {root}')
    if not (BASE / 'ready.json').exists():
        raise FileNotFoundError(BASE / 'ready.json')
    root.mkdir(parents=True)
    source = root / 'source'
    for directory in ('genrec', 'scripts', 'tests'):
        for path in (REPO / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.json'):
                target = source / path.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    experiment = Path('experiments/verifier_residual_20260908')
    shutil.copytree(REPO / experiment, source / experiment,
                    ignore=shutil.ignore_patterns('__pycache__'))
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    base_ready = json.loads((BASE / 'ready.json').read_text())
    manifest = {
        'source_files': files,
        'base_cache': str(BASE),
        'base_ready_sha256': digest(BASE / 'ready.json'),
        'base_metadata_sha256': digest(BASE / 'metadata.json'),
        'base_array_hashes': base_ready['array_hashes'],
    }
    (root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(root)


if __name__ == '__main__':
    main()
