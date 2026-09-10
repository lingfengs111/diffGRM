#!/usr/bin/env python3
"""Create a small, immutable experiment record for Git synchronization."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sem_ids",
    ".sent_emb",
}
RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MACHINE_ID_RE = RECORD_ID_RE
STATUSES = ("planned", "running", "complete", "failed", "invalid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_file(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"not a file: {raw}")
    return path


def parse_named_file(spec: str) -> tuple[Path, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"expected NAME=PATH, got: {spec}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"artifact name must be a safe relative path: {name}")
    return relative, resolve_file(raw_path)


def validate_metrics(path: Path, record_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("metrics.json must use schema_version 1")
    if payload.get("study_id") != record_id:
        raise ValueError("metrics.json study_id must equal --record-id")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("metrics.json rows must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"metrics row {index} must be an object")
        for key in ("dataset", "split", "method", "metrics"):
            if key not in row:
                raise ValueError(f"metrics row {index} is missing {key}")
        if not isinstance(row["metrics"], dict):
            raise ValueError(f"metrics row {index}.metrics must be an object")
        for name, value in row["metrics"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"metric {name!r} in row {index} is not numeric")


def copy_evidence(
    source: Path,
    destination: Path,
    record_root: Path,
    role: str,
    max_bytes: int,
) -> dict[str, Any]:
    size = source.stat().st_size
    if source.suffix.lower() in BLOCKED_SUFFIXES:
        raise ValueError(
            f"refusing model/array artifact {source}; record it with --checkpoint"
        )
    if size > max_bytes:
        raise ValueError(
            f"artifact {source} is {size} bytes (limit {max_bytes}); keep it local"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "role": role,
        "source": display_path(source),
        "stored_as": destination.relative_to(record_root).as_posix(),
        "bytes": size,
        "sha256": sha256(destination),
    }


def checkpoint_reference(raw: str, hash_checkpoint: bool) -> dict[str, Any]:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    entry: dict[str, Any] = {"path": display_path(path), "exists": path.is_file()}
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        if hash_checkpoint:
            entry["sha256"] = sha256(path)
    return entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True)
    parser.add_argument(
        "--machine-id",
        help=(
            "producer directory below experiment_records/incoming; defaults "
            "to git config diffgrm.machine, then the short hostname"
        ),
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--status", required=True, choices=STATUSES)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--protocol", default="")
    parser.add_argument("--command", default=None, help="exact launch command")
    parser.add_argument("--summary", help="Markdown findings copied as summary.md")
    parser.add_argument("--metrics", help="normalized schema-v1 JSON copied as metrics.json")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="small evidence file copied below artifacts/; repeat as needed",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="resolved/source config copied below configs/; repeat as needed",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="checkpoint reference only; file is never copied",
    )
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="compute SHA-256 for checkpoint references (can be slow)",
    )
    parser.add_argument(
        "--max-artifact-bytes", type=int, default=5 * 1024 * 1024
    )
    parser.add_argument(
        "--records-root", default="experiment_records", help=argparse.SUPPRESS
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not RECORD_ID_RE.fullmatch(args.record_id):
        parser.error("--record-id may contain only letters, numbers, dot, dash, underscore")
    if args.max_artifact_bytes <= 0:
        parser.error("--max-artifact-bytes must be positive")

    machine_id = args.machine_id or git_text("config", "--get", "diffgrm.machine")
    if not machine_id:
        machine_id = socket.gethostname().split(".", 1)[0]
    if not MACHINE_ID_RE.fullmatch(machine_id):
        parser.error(
            "machine id may contain only letters, numbers, dot, dash, underscore; "
            "set it with git config diffgrm.machine NAME"
        )

    records_root = Path(args.records_root).expanduser()
    if not records_root.is_absolute():
        records_root = REPO_ROOT / records_root
    records_root = records_root.resolve()
    for manifest_path in records_root.rglob("manifest.json"):
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot inspect existing record {manifest_path}: {error}")
        if existing.get("record_id") == args.record_id:
            parser.error(
                f"record_id already exists and is immutable: {manifest_path.parent}"
            )

    producer_root = records_root / "incoming" / machine_id
    producer_root.mkdir(parents=True, exist_ok=True)
    destination = producer_root / args.record_id

    staging = producer_root / f".{args.record_id}.tmp-{os.getpid()}"
    files: list[dict[str, Any]] = []
    try:
        staging.mkdir(parents=True, exist_ok=False)
        if args.summary:
            source = resolve_file(args.summary)
            files.append(
                copy_evidence(
                    source,
                    staging / "summary.md",
                    staging,
                    "summary",
                    args.max_artifact_bytes,
                )
            )
        if args.metrics:
            source = resolve_file(args.metrics)
            validate_metrics(source, args.record_id)
            files.append(
                copy_evidence(
                    source,
                    staging / "metrics.json",
                    staging,
                    "metrics",
                    args.max_artifact_bytes,
                )
            )
        for spec in args.artifact:
            name, source = parse_named_file(spec)
            files.append(
                copy_evidence(
                    source,
                    staging / "artifacts" / name,
                    staging,
                    "artifact",
                    args.max_artifact_bytes,
                )
            )
        for spec in args.config:
            name, source = parse_named_file(spec)
            files.append(
                copy_evidence(
                    source,
                    staging / "configs" / name,
                    staging,
                    "config",
                    args.max_artifact_bytes,
                )
            )

        status_text = git_text("status", "--porcelain")
        manifest = {
            "schema_version": 1,
            "record_id": args.record_id,
            "title": args.title,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": args.status,
            "datasets": sorted(set(args.dataset)),
            "tags": sorted(set(args.tag)),
            "protocol": args.protocol,
            "command": args.command,
            "recorded_by": {
                "machine_id": machine_id,
                "host": socket.gethostname(),
                "user": os.environ.get("USER") or getpass.getuser(),
            },
            "git_snapshot": {
                "commit": git_text("rev-parse", "HEAD"),
                "branch": git_text("branch", "--show-current"),
                "dirty": bool(status_text),
            },
            "files": files,
            "checkpoints": [
                checkpoint_reference(path, args.hash_checkpoints)
                for path in args.checkpoint
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if staging.exists():
            shutil.rmtree(staging)
        parser.error(str(error))

    print(f"created {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
