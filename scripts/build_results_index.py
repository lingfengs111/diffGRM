#!/usr/bin/env python3
"""Validate normalized experiment metrics and build a deterministic Markdown index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_METRICS = (
    ("ndcg@5", "NDCG@5"),
    ("recall@5", "Recall@5"),
    ("ndcg@10", "NDCG@10"),
    ("recall@10", "Recall@10"),
    ("candidate_recall@72", "Candidate R@72"),
)


def compact(value: Any) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ").strip()
    return text


def metric_text(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {key!r} must be numeric")
    return f"{value:.6f}"


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifests: dict[Path, dict[str, Any]] = {}
    seen_record_ids: dict[str, Path] = {}
    for manifest_path in sorted(root.rglob("manifest.json")):
        relative_parts = manifest_path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        record_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported manifest schema: {manifest_path}")
        record_id = manifest.get("record_id")
        if record_id != record_dir.name:
            raise ValueError(f"manifest record_id does not match directory: {record_dir}")
        relative_record = record_dir.relative_to(root)
        parts = relative_record.parts
        if not (
            (len(parts) == 2 and parts[0] == "archive")
            or (len(parts) == 3 and parts[0] == "incoming")
        ):
            raise ValueError(
                f"record must be under archive/<id> or incoming/<machine>/<id>: "
                f"{record_dir}"
            )
        previous = seen_record_ids.get(record_id)
        if previous is not None:
            raise ValueError(
                f"duplicate record_id {record_id!r}: {previous} and {record_dir}"
            )
        seen_record_ids[record_id] = record_dir
        manifests[record_dir] = manifest

    for metrics_path in sorted(root.rglob("metrics.json")):
        relative_parts = metrics_path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        record_dir = metrics_path.parent
        manifest_path = record_dir / "manifest.json"
        manifest = manifests.get(record_dir)
        if manifest is None:
            raise ValueError(f"missing manifest: {manifest_path}")
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_payload.get("schema_version") != 1:
            raise ValueError(f"unsupported metrics schema: {metrics_path}")
        record_id = manifest.get("record_id")
        if metrics_payload.get("study_id") != record_id:
            raise ValueError(f"metrics study_id does not match manifest: {metrics_path}")
        relative_record = record_dir.relative_to(root)
        parts = relative_record.parts
        if parts and parts[0] == "archive":
            location = "archive"
        elif len(parts) >= 3 and parts[0] == "incoming":
            location = f"incoming/{parts[1]}"
        else:
            location = "legacy-root"
        rows = metrics_payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"rows must be a list: {metrics_path}")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"row {index} is not an object: {metrics_path}")
            for key in ("dataset", "split", "method", "metrics"):
                if key not in row:
                    raise ValueError(f"row {index} missing {key}: {metrics_path}")
            records.append(
                {
                    "record_id": record_id,
                    "has_summary": (record_dir / "summary.md").is_file(),
                    "title": manifest.get("title", record_id),
                    "date": manifest.get("date", manifest.get("created_at", "")[:10]),
                    "status": manifest.get("status", "unknown"),
                    "location": location,
                    "record_path": relative_record.as_posix(),
                    **row,
                }
            )
    return records


def render(records: list[dict[str, Any]]) -> str:
    metric_headers = [label for _, label in COMMON_METRICS]
    header = [
        "Study", "Date", "Status", "Location", "Dataset", "Split", "Method",
        *metric_headers, "Notes",
    ]
    lines = [
        "# Experiment result index",
        "",
        "Generated deterministically by `python scripts/build_results_index.py`.",
        "Rows are comparable only when their protocol notes agree.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * 7 + ["---:"] * len(COMMON_METRICS) + ["---"]) + "|",
    ]
    order = sorted(
        records,
        key=lambda row: (
            compact(row["date"]),
            compact(row["record_id"]),
            compact(row["dataset"]),
            compact(row["method"]),
        ),
    )
    for row in order:
        metrics = row["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics must be an object in {row['record_id']}")
        record_target = f"{compact(row['record_path'])}/" + (
            "summary.md" if row["has_summary"] else "manifest.json"
        )
        study = f"[{compact(row['record_id'])}]({record_target})"
        cells = [
            study,
            compact(row["date"]),
            compact(row["status"]),
            compact(row["location"]),
            compact(row["dataset"]),
            compact(row["split"]),
            compact(row["method"]),
            *[metric_text(metrics, key) for key, _ in COMMON_METRICS],
            compact(row.get("notes", "")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if INDEX.md is stale")
    parser.add_argument("--records-root", default="experiment_records", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = Path(args.records_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    output = root / "INDEX.md"
    try:
        rendered = render(load_records(root))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.check:
        existing = output.read_text(encoding="utf-8") if output.is_file() else ""
        if existing != rendered:
            print(f"stale: {output}", file=sys.stderr)
            return 1
        print(f"up to date: {output}")
        return 0

    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
