# Experiment records

This directory is intentionally tracked even though `runs/` is ignored. It is
the small, reviewable evidence layer shared between machines.

Use one unique directory per experiment or tightly coupled suite. A routine
record contains `manifest.json`, `metrics.json`, and optionally small source
files in `artifacts/`. `summary.md` is optional: add it for a completed suite,
a conclusion-changing result, or an important validity/protocol note—not for
every additional seed. Do not place checkpoints, embeddings, rank matrices,
full logs, or datasets here.

`metrics.json` schema version 1 is:

```json
{
  "schema_version": 1,
  "study_id": "20260908_example",
  "rows": [
    {
      "dataset": "Video23",
      "split": "test",
      "method": "descriptive method name",
      "metrics": {
        "ndcg@5": 0.0,
        "recall@5": 0.0,
        "ndcg@10": 0.0,
        "recall@10": 0.0,
        "candidate_recall@72": 0.0
      },
      "notes": "L20, collision-free item-level, seed 2026"
    }
  ]
}
```

Unknown metrics may be added to `metrics`; the index displays the common five.
Do not use `0` for a metric that was not measured—omit it. Method names and
notes must state protocol differences that would make rows incomparable.

Create new records with `scripts/record_experiment.py`, then regenerate the
deterministic common table with `scripts/build_results_index.py`. See
`docs/EXPERIMENT_SYNC.md` for the complete two-machine workflow.
