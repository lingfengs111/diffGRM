#!/usr/bin/env python3
"""Compare two DiffGRM evaluation JSON files for a history-length ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_METRICS = (
    "ndcg@5",
    "ndcg@10",
    "recall@5",
    "recall@10",
    "weighted_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", default="L50")
    parser.add_argument("--candidate-label", default="L20")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)

    comparisons: dict[str, dict[str, float | None]] = {}
    for metric in PRIMARY_METRICS:
        baseline_value = float(baseline[metric])
        candidate_value = float(candidate[metric])
        delta = candidate_value - baseline_value
        comparisons[metric] = {
            args.baseline_label: baseline_value,
            args.candidate_label: candidate_value,
            "absolute_delta": delta,
            "relative_delta_pct": (
                100.0 * delta / baseline_value if baseline_value else None
            ),
        }

    result = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_file": str(args.baseline.resolve()),
        "candidate_file": str(args.candidate.resolve()),
        "comparison": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    markdown_path = args.output.with_suffix(".md")
    rows = [
        f"# {args.candidate_label} versus {args.baseline_label} history-length ablation",
        "",
        f"| Metric | {args.baseline_label} | {args.candidate_label} | Absolute delta | Relative delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in comparisons.items():
        relative = values["relative_delta_pct"]
        relative_text = "n/a" if relative is None else f"{relative:+.2f}%"
        rows.append(
            f"| {metric} | {values[args.baseline_label]:.6f} | "
            f"{values[args.candidate_label]:.6f} | "
            f"{values['absolute_delta']:+.6f} | {relative_text} |"
        )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Markdown comparison: {markdown_path}")


if __name__ == "__main__":
    main()
