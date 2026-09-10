#!/usr/bin/env python3
"""Promote terminal machine-owned experiment records into the shared archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TERMINAL_STATUSES = {"complete", "failed", "invalid"}


def git_machine_id() -> str | None:
    result = subprocess.run(
        ["git", "config", "--get", "diffgrm.machine"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_id", nargs="+")
    parser.add_argument(
        "--machine-id",
        help="source directory; defaults to git config diffgrm.machine or hostname",
    )
    parser.add_argument("--records-root", default="experiment_records", help=argparse.SUPPRESS)
    args = parser.parse_args()

    machine_id = args.machine_id or git_machine_id() or socket.gethostname().split(".", 1)[0]
    if not ID_RE.fullmatch(machine_id):
        parser.error(f"invalid machine id: {machine_id}")
    if any(not ID_RE.fullmatch(record_id) for record_id in args.record_id):
        parser.error("record ids may contain only letters, numbers, dot, dash, underscore")

    root = Path(args.records_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    archive = root / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    moves: list[tuple[Path, Path]] = []
    for record_id in args.record_id:
        source = root / "incoming" / machine_id / record_id
        destination = archive / record_id
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            parser.error(f"missing incoming record: {source}")
        if destination.exists():
            parser.error(f"archive destination already exists: {destination}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("record_id") != record_id:
            parser.error(f"manifest id mismatch: {manifest_path}")
        status = manifest.get("status")
        if status not in TERMINAL_STATUSES:
            parser.error(
                f"record {record_id} has non-terminal status {status!r}; "
                "archive only complete, failed, or invalid records"
            )
        moves.append((source, destination))

    for source, destination in moves:
        shutil.move(str(source), str(destination))
        print(f"archived {source.relative_to(root)} -> {destination.relative_to(root)}")
    print("rebuild INDEX.md, then stage the moves with git add -A experiment_records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
