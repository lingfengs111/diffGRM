# Two-machine experiment synchronization

The Git repository is the shared **code and evidence layer**, not a mirror of
the complete training filesystem. Raw datasets, checkpoints, caches, rank
arrays, and logs stay local. Source code, experiment configurations, concise
findings, normalized metrics, and small result JSON files are committed.

## What belongs in Git

- `genrec/`, `scripts/`, `experiments/`, and `tests/`;
- root configuration and handoff documents;
- `docs/` for current protocols and project-level conclusions;
- `experiment_records/` for one immutable directory per completed, failed, or
  deliberately retained running experiment.

Do not commit `runs/`, `saved/`, `data/`, `cache/`, `logs/`, model weights, or
large per-example arrays. `.gitignore` enforces the common cases. A checkpoint
may be named in a record, with size and optionally a SHA-256 digest, without
copying the checkpoint itself.

## Record contract

Each record directory contains:

- `manifest.json`: identity, status, dataset, protocol, source paths, Git state,
  and hashes of copied evidence;
- `metrics.json`: normalized rows used to build the common index;
- optional `summary.md`: interpretation, validity qualifications, and next
  decision when a run changes the research conclusion;
- optional small files under `artifacts/`, such as the original `result.json`,
  exact command, resolved config, or statistical report.

For an ordinary additional seed or routine control, `manifest.json` plus
`metrics.json` and any essential small raw result are enough. Do not manufacture
a new prose summary for every run. Write one when a suite closes, a protocol or
validity issue needs explanation, or the project-level conclusion changes.

Statuses are `planned`, `running`, `complete`, `failed`, or `invalid`. Never
silently replace an invalid result with a rerun: retain the invalid path/status
and give the valid rerun a distinct name. A record is a snapshot, so a running
study should receive a new revision or an explicit reviewed update when it
finishes.

Use a globally unique ID such as
`20260908_video23_rank32_server_b_seed2026`. Independent record directories
make ordinary Git merges much easier than having both machines append to one
CSV or edit the same scoreboard.

Create a record with the standard helper:

```bash
python scripts/record_experiment.py \
  --record-id 20260908_video23_rank32_server_b_seed2026 \
  --title "Video23 rank-32 listwise, seed 2026" \
  --status complete \
  --dataset Video23 \
  --tag rank32 \
  --summary runs/example/FINAL_FINDINGS.md \
  --metrics runs/example/metrics_for_record.json \
  --artifact result.json=runs/example/result.json \
  --config train.yaml=experiments/example/train.yaml \
  --checkpoint runs/example/best.pt
python scripts/build_results_index.py
```

`metrics_for_record.json` follows the short schema documented in
`experiment_records/README.md`. The helper refuses large artifacts and model
weight files. It records checkpoints by reference only.

Before committing, run:

```bash
python scripts/build_results_index.py --check
python scripts/record_experiment.py --help
git status --short
```

## GitHub setup

This directory is already a Git repository whose `origin` points to the public
DiffGRM source. Do not run `git init`, and do not overwrite `origin`. Create an
**empty private GitHub repository** (no generated README, license, or
`.gitignore`), then add it as a second remote:

```bash
git remote add research git@github.com:OWNER/PRIVATE_REPO.git
git push -u research main
```

The initial push should happen only after the current dirty working tree has
been reviewed and committed. Until that review, neither machine should treat
the public `origin/main` commit as a reproducible research snapshot.

## Daily workflow

Give each machine or experiment its own branch, for example
`machine-a/tokenizer-controls` and `machine-b/rank32`. At the start of work,
fetch `research` and branch from the agreed integration branch. At the end of a
run, create its record, rebuild the index, commit only the new/changed
code/config/record files, and push that branch. Unchanged historical files are
content-addressed by Git and are not uploaded again. Merge through GitHub or on
one designated integration machine.

Avoid having both machines push directly to `main`: source-code conflicts need
review, while distinct experiment-record directories usually merge cleanly.
Checkpoint transfer is a separate operation (manual copy, `rsync`, or object
storage); verify a shared checkpoint by SHA-256 before resuming or evaluating
it.

Do not infer run liveness from a sandboxed `ps` view alone. Prefer a combination
of an advancing `progress.json`/log timestamp, the recorded launcher PID on the
host, and GPU utilization. The OPQ4-Hungarian control on 2026-09-07 was a real
example: its host process was hidden from one process namespace while its
progress file continued to advance.
