#!/usr/bin/env python3
"""Show experiment records, optionally restricted to changes since a Git ref."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def changed_record_paths(root: Path, since: str) -> set[Path]:
    result = subprocess.run(
        [
            "git", "diff", "--name-only", "--diff-filter=ACMR",
            f"{since}..HEAD", "--", root.relative_to(REPO_ROOT).as_posix(),
        ],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"cannot resolve Git ref {since!r}")
    changed = set()
    for line in result.stdout.splitlines():
        path = REPO_ROOT / line
        try:
            changed.add(path.relative_to(root))
        except ValueError:
            continue
    return changed


def record_changed(relative_record: Path, changed: set[Path]) -> bool:
    prefix = relative_record.parts
    return any(path.parts[: len(prefix)] == prefix for path in changed)


def metric_text(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return "-" if value is None else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        help="show only records changed between this Git ref and HEAD; use ORIG_HEAD after pull",
    )
    parser.add_argument("--machine", help="filter by producer machine id or hostname")
    parser.add_argument("--status", help="filter by manifest status")
    parser.add_argument("--records-root", default="experiment_records", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = Path(args.records_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    changed = changed_record_paths(root, args.since) if args.since else None

    records = []
    seen = {}
    for manifest_path in sorted(root.rglob("manifest.json")):
        relative = manifest_path.parent.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if changed is not None and not record_changed(relative, changed):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_id = manifest.get("record_id", manifest_path.parent.name)
        if record_id in seen:
            raise ValueError(f"duplicate record id {record_id}: {seen[record_id]} and {relative}")
        seen[record_id] = relative
        producer = manifest.get("recorded_by", {}).get("machine_id") or manifest.get(
            "recorded_by", {}
        ).get("host", "unknown")
        if args.machine and producer != args.machine:
            continue
        if args.status and manifest.get("status") != args.status:
            continue
        metrics_path = manifest_path.parent / "metrics.json"
        rows = []
        if metrics_path.is_file():
            rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
        records.append((manifest, relative, producer, rows))

    if not records:
        print("No matching experiment-record updates.")
        return 0

    for manifest, relative, producer, rows in records:
        print(
            f"[{manifest.get('status', 'unknown')}] {manifest.get('record_id')} "
            f"({relative}; producer={producer})"
        )
        print(f"  {manifest.get('title', '')}")
        for row in rows:
            metrics = row.get("metrics", {})
            print(
                "  - "
                f"{row.get('dataset')}/{row.get('split')}: {row.get('method')} | "
                f"NDCG@10={metric_text(metrics, 'ndcg@10')} "
                f"Recall@10={metric_text(metrics, 'recall@10')} "
                f"CandidateR@72={metric_text(metrics, 'candidate_recall@72')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
