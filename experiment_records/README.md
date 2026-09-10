# Experiment records

This directory is intentionally tracked even though `runs/` is ignored. It is
the small, reviewable evidence layer shared between machines.

The directory has two explicit lifecycle layers:

```text
experiment_records/
  incoming/<machine-id>/   # owned and written by exactly one machine
  archive/<record-id>/     # terminal, reviewed, shared, immutable records
  INDEX.md                 # generated view across both layers
```

Use one unique record directory per experiment or tightly coupled suite. A
routine record contains `manifest.json`, `metrics.json`, and optionally small
source files in `artifacts/`. `summary.md` is optional: add it for a completed
suite, a conclusion-changing result, or an important validity/protocol note.
Do not place checkpoints, embeddings, rank matrices, full logs, or datasets
here.

Configure a stable producer ID once in every clone:

```bash
git config --local diffgrm.machine primary    # original workstation
git config --local diffgrm.machine gpu-ckpt   # checkpoint-capable server
git config --local diffgrm.machine gpu-git    # GitHub-only server
```

`record_experiment.py` then writes automatically to that machine's incoming
directory. A machine may copy `incoming/STATUS_TEMPLATE.md` to
`incoming/<machine-id>/STATUS.md` for its mutable queue/status overview. It
must not edit another machine's directory.

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

Create new records with `scripts/record_experiment.py`. After review/merge,
promote terminal records with `scripts/archive_experiment_records.py`; then
regenerate the deterministic common table with `scripts/build_results_index.py`.

After a pull, `python scripts/show_experiment_updates.py --since ORIG_HEAD`
prints only the records and metrics introduced by that pull. See
`docs/EXPERIMENT_SYNC.md` for the complete multi-machine workflow.
