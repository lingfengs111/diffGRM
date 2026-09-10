# Structured Generative Recommendation Research Fork

This repository started from DiffGRM, but the active research line is now a
**collision-free one-pass structured semantic retriever with an independent
causal AR verifier**. The `DiffGRM` repository and `DIFF_GRM` package names are
retained for continuity; the current main method is not iterative diffusion.

The primary experimental family is **Amazon Reviews 2023**, using the following
three domains:

| Short name | Amazon23 domain | Train | Validation | Test | Items |
|---|---|---:|---:|---:|---:|
| Video23 | Video Games | 530,300 | 94,762 | 94,762 | 25,612 |
| Music23 | Musical Instruments | 339,519 | 57,439 | 57,439 | 24,587 |
| Science23 | Industrial and Scientific | 259,992 | 50,985 | 50,985 | 25,848 |

Office23 is prepared as a possible larger-scale extension, but it is not one
of the three current primary domains. Amazon14 experiments are retained only
as legacy reproduction and literature-comparison controls.

## Start here

- [Current research status](docs/CURRENT_STATUS_2026-09-10.md): the current
  method, latest conclusions, and protocol distinctions.
- [Research handoff](RESEARCH_HANDOFF_2026-09-01.md): architecture, code map,
  canonical results, and diagnostic findings.
- [Experiment result index](experiment_records/INDEX.md): lightweight,
  Git-synchronized numerical records.
- [Multi-machine synchronization](docs/EXPERIMENT_SYNC.md): directory
  ownership, archive lifecycle, and what belongs in Git
  and how to publish a new experiment without committing checkpoints.
- [Amazon23 data/configuration guide](experiments/amazon23_domains/README.md):
  prepared split statistics, audits, and baseline launch commands.

## Current system

The main pipeline has three components:

1. Item text is embedded with Sentence-T5 and quantized into semantic IDs.
   Formal catalogs are repaired to be injective, so evaluation always resolves
   a prediction to one concrete item.
2. A one-forward history encoder and parallel coordinate heads score the legal
   item catalog using unary plus learned pairwise tuple compatibility, then
   retrieve Top-K candidates (normally K=72).
3. A separately trained causal AR model teacher-forces each proposed complete
   SID, scores the paths in parallel, and fuses its standardized scores with
   the drafter scores using a validation-selected alpha.

New Amazon23 experiments use the latest 20 history items (`L20`) unless an
exception is explicitly labeled. Do not mix the historical Video23 L50 rows
with the current L20 comparison. See the
[history-length protocol](experiments/AMAZON_MAXLEN_PROTOCOL.md).

## Current Amazon23 result snapshot

All rows below are full test, collision-free concrete-item results. They are
orientation numbers rather than a final multi-seed paper table.

| Dataset/protocol | Method | NDCG@10 | Recall@10 |
|---|---|---:|---:|
| Video23 L20 | NDCG-weighted DPO, four-seed mean | 0.048258 | 0.090820 |
| Video23 L20 | pooled one-pass pairwise + AR, fused-checkpoint control | 0.048461 | 0.091281 |
| Video23 L20 | local Latte, beam 500 | 0.051082 | 0.095407 |
| Music23 L20 | RQ-KMeans3 drafter + matched AR | 0.032319 | **0.061474** |
| Music23 L20 | RQ-KMeans3 drafter + OPQ4 AR | **0.032320** | 0.060516 |
| Science23 L20 | OPQ4 drafter + RQ-KMeans3 AR dual view | **0.024879** | **0.047230** |
| Video23 L50, historical | DiffGRM-init one-pass pairwise + AR | 0.048585 | 0.091123 |

The NDCG-weighted DPO row improves its fixed candidate-selected baseline in all
four seeds, but it has not yet been applied to the stronger fused-checkpoint
drafter control. The result index records the individual seeds, additional
baselines, negative results, tokenizer controls, and protocol notes.

## Environment

```bash
git clone git@github.com:lingfengs111/diffGRM.git
cd diffGRM
conda create -n diffgrm python=3.10 -y
conda activate diffgrm
pip install -r requirements.txt
```

The current long-running environment on the original machine is
`/home/lingfengs111/miniconda3/envs/diffgrm/bin/python`. A second machine may
use a different path; update `python_bin` in older launch scripts accordingly.

## Amazon23 data

Datasets are deliberately not stored in Git. Each prepared domain directory
contains `item_vocab.csv`, `item_texts.csv`, and the CleanGR train/validation/
test JSONL splits. Update `data_dir` after copying data to the new machine:

- `experiments/amazon23_domains/video23.yaml`
- `experiments/amazon23_domains/music23.yaml`
- `experiments/amazon23_domains/science23.yaml`

The committed files currently contain paths from the original workstation, so
this path adjustment is required when the directory layout differs.

Audit a copied dataset before training:

```bash
python scripts/validate_prepared_protocol.py \
  --dataset AmazonReviews2023CleanGR \
  --config experiments/amazon23_domains/common.yaml \
  --config experiments/amazon23_domains/video23.yaml \
  --data-only
```

Replace `video23.yaml` with `music23.yaml` or `science23.yaml` as needed.

## Main code and experiment entry points

- `scripts/train_parallel_opq_drafter.py`: current one-pass structured drafter
  training, candidate retrieval, AR reranking, and fusion evaluation.
- `genrec/models/DIFF_GRM/parallel_drafter.py`: unary/pairwise catalog scorer.
- `genrec/models/DIFF_GRM/encoder_head_drafter.py`: random encoder plus parallel
  coordinate heads.
- `genrec/models/AR_GRM/model.py`: standalone generation and exact
  teacher-forced candidate-path scoring.
- `experiments/music23_transfer/`: Music23 AR, DiffGRM, SASRec, and current
  one-pass transfer launchers.
- `experiments/science23_transfer/`: Science23 L20 transfer launchers.
- `experiments/tokenizer_controls_20260907/`: Science23 tokenizer, collision
  repair, and dual-view controls.
- `experiments/history_interest_20260907/`: completed Science23/Video23 L20
  history-reader and multiple-interest suite.
- `experiments/verifier_objectives_20260909/`: rank-aware AR objectives and the
  four-seed NDCG-weighted DPO follow-up.
- `experiments/atomic_sid_controls_20260909/`: matched atomic-table, OPQ, and
  completely SID-free controls.
- `experiments/verifier_next_20260909/`: proposal-aware adaptation, generation
  union, and verifier-capacity controls.

Several historical launchers contain absolute repository, Python, checkpoint,
or dataset paths. Review these variables before running them on another
machine. Training artifacts go under ignored `runs/` and `saved/` directories.

## Recording a new result

Routine experiments do not need a new prose summary. Configure this clone's
machine ID once, then record metrics and provenance into its machine-owned
incoming directory:

```bash
git config --local diffgrm.machine MACHINE_ID
python scripts/record_experiment.py --help
git status --short
```

A `summary.md` is useful only when a suite closes, a validity issue is found,
or the research conclusion changes. The integration step moves terminal
records into the shared archive and rebuilds the index. Checkpoints and large
rank arrays remain local and are referenced rather than copied.

## Legacy Amazon14 scope

The original DiffGRM Sports/Beauty/Toys work and later Beauty14/RPG comparison
audits remain available under `experiments/amazon14_domains/` and related
experiment directories. They are useful for reproducing published tables and
studying SID-collision inflation, but they are no longer the default project
entry point or the primary dataset family for new method development.
