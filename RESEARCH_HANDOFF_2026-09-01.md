# CleanGR / structured drafter + AR research handoff (2026-09-01)

> **Update (2026-09-07):** this remains the detailed architecture handoff, but
> its result snapshot predates the L20 Science23/Video23 history-interest and
> tokenizer-control suites. Read
> [`docs/CURRENT_STATUS_2026-09-07.md`](docs/CURRENT_STATUS_2026-09-07.md) for
> the current result-level amendments. In particular, do not mix the canonical
> historical Video23 L50 table below with the newer Video23 L20 suite.

This is the current handoff for the active research line.  It supersedes
`runs/canonical_full/progress_summary_2026-08-23.md`, which is useful as
history but predates the one-pass drafter, parameter-fairness controls, and
Music23 transfer.

## What the project is trying to establish

The current question is no longer simply whether masked diffusion can replace
autoregressive decoding.  The strongest system has three stages:

1. **Tokenizer:** Sentence-T5 item text embeddings, PCA-256, collision-free
   OPQ4/PQ semantic IDs (four coordinates, codebook size 256).  Hungarian
   repair makes the final catalog mapping injective.
2. **Drafter/retriever:** a one-forward history encoder with four parallel SID
   heads.  It scores every legal catalog tuple with unary coordinate evidence
   plus learned pairwise coordinate compatibility and returns Top-K items.
3. **Verifier:** a separately trained AR encoder-decoder teacher-forces each
   complete proposed SID and sums its four conditional log-probabilities.
   Per-query standardized drafter and AR scores are fused; alpha is selected
   only on validation (usually 0.75).

The current one-pass drafter is best described as a **parallel structured
semantic retriever**, not as iterative diffusion.  DiffGRM is an important
baseline and optional initialization, but denoising initialization is not
required: a random-initialized encoder + four heads performs strongly, and on
Music23 it is slightly better than the DiffGRM-initialized control.

The next research target is to explain and exploit the behavioral
complementarity between the independently trained drafter and AR verifier.

## Non-negotiable evaluation protocol

- Formal metrics must resolve a generated SID to a **concrete item**.  Do not
  count a collision bucket as correct merely because its SID matches.
- Current OPQ catalogs are globally collision-free after Hungarian repair.
- One target item exists per example, so Hit@K equals Recall@K here.
- Always report NDCG and Recall at both 5 and 10.
- Smoke/subset experiments are for debugging only.  Paper claims require full
  train/validation/test splits.
- Music23 and future Amazon23 domains use history length L20.  Existing
  Video23 results use a historical L50 protocol and must be labeled as such.
- Preserve the dirty working tree.  Many active changes are uncommitted; do
  not reset, checkout, clean, or overwrite unrelated files.

## Data and environment

- Repository: `/home/lingfengs111/codes/GR_variant/DiffGRM`
- Python: `/home/lingfengs111/miniconda3/envs/diffgrm/bin/python`
- Video23 full core-5:
  `/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_video_game/raw_core5`
  (530,300 train / 94,762 validation / 94,762 test / 25,612 items, L50)
- Music23 full core-5:
  `/home/lingfengs111/codes/GR/CleanGR/outputs/amazon23_music/raw_core5`
  (339,519 train / 57,439 validation / 57,439 test / 24,587 items, L20)
- Shared Amazon23 configuration:
  `experiments/amazon23_domains/common.yaml`
- Video23 formal configuration:
  `experiments/canonical_full/video23_cf_official.yaml`
- Music23 formal configuration:
  `experiments/music23_transfer/music23_l20_long.yaml`

## Code map

Read these first:

1. `scripts/train_parallel_opq_drafter.py`
   - training/evaluation entry point for one-pass drafter;
   - legal-catalog CE and sampled-catalog CE;
   - proposal Top-K, AR reranking, score normalization and fusion.
2. `genrec/models/DIFF_GRM/parallel_drafter.py`
   - catalog unary scoring;
   - low-rank pairwise/triple compatibility modules.
3. `genrec/models/DIFF_GRM/encoder_head_drafter.py`
   - random-initialized encoder + four parallel output heads.
4. `genrec/models/AR_GRM/model.py`
   - standalone AR generation;
   - `score_candidate_paths` exact teacher-forced candidate scoring;
   - cached history and cross-attention K/V implementation.
5. `scripts/train_candidate_aware_verifier.py`
   - learned verifier and residual-over-AR experiments.
6. `scripts/train_shared_encoder_ar_verifier.py`
   - shared-history-encoder control.
7. `scripts/train_ann_drafter.py`
   - Exact-MIPS/ANN controls.
8. `scripts/evaluate_opq_subset_oracle.py`
   - earlier OPQ coordinate/subset oracle diagnostics.

## Canonical and recent results

All numbers below are test NDCG@10 / Recall@10.

### Video23 full, historical L50

| Method | NDCG@10 | Recall@10 |
|---|---:|---:|
| SASRec full softmax | 0.047593 | 0.087208 |
| Collision-free guided DiffGRM direct | 0.043537 | 0.081657 |
| OPQ + standalone constrained AR | 0.044655 | 0.084148 |
| Exact-MIPS ID drafter + AR fusion | 0.046711 | 0.088284 |
| Random encoder + four heads + pairwise + AR fusion | 0.048009 | 0.090289 |
| DiffGRM-initialized pairwise drafter + AR fusion | **0.048585** | **0.091123** |
| Shared-encoder AR control | 0.045841 | 0.085942 |

The shared-encoder, non-AR, bidirectional/MLP, and lightweight distilled
verifier controls are all below the independent AR fusion.  This is evidence
for independent specialization, but it does not yet explain what each model
learns.

Important sources:

- `runs/canonical_full/video23_paper_scoreboard.md`
- `runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/result.json`
- `runs/diffusion_necessity/video23_full_20260829/encoder4_four_head_pairwise_r51/result.json`
- `runs/ann_drafter/video23_exact_mips_id_v1/result.json`
- `runs/parameter_fairness/video23_full_20260830/shared_encoder_ar/result.json`
- `runs/verifier_only/video23_full_20260830/`

### Music23 full, canonical L20

| Method | NDCG@10 | Recall@10 |
|---|---:|---:|
| SASRec full softmax | 0.030341 | 0.055990 |
| Collision-free guided DiffGRM direct | 0.027057 | 0.051376 |
| OPQ + standalone constrained AR | 0.028986 | 0.054562 |
| Random one-pass drafter only | 0.029519 | 0.055694 |
| AR-only reranking of the same 72 proposals | 0.029827 | 0.056234 |
| Random one-pass + AR fusion | **0.032250** | **0.059855** |
| DiffGRM-pretrained one-pass + AR fusion | 0.031912 | 0.059785 |

The Music result is a clean cross-domain confirmation that fusion helps and
that DiffGRM denoising initialization is not necessary.  The random one-pass
result is in:
`runs/music23_transfer/random_encoder4_pairwise_ar/result.json`.
The completed pretrained control is in:
`runs/music23_transfer/diff_pretrained_masked_pairwise_ar/result.json`.

### Video23 candidate budget, frozen half-size model

| K | candidate Recall@K | fusion NDCG@10 | fusion Recall@10 | ms/example |
|---:|---:|---:|---:|---:|
| 10 | 0.082480 | 0.045561 | 0.082480 | 0.940 |
| 32 | 0.159864 | 0.047939 | 0.090036 | 0.947 |
| 64 | 0.222262 | 0.047979 | 0.090194 | 1.092 |
| 72 | 0.234387 | 0.047991 | 0.090226 | 1.074 |
| 128 | 0.299213 | 0.048103 | 0.090374 | 1.246 |

K=32 retains almost all final Top-10 quality despite much lower candidate
coverage.  Correct targets at drafter ranks 33--128 are rarely promoted by the
current AR verifier.  This is a central diagnostic clue, not merely an
efficiency result.

Source: `runs/capacity_fairness/video23_half_2x2_d176/candidate_budget/`.

### Sampled catalog CE and residual verifier

- Half-size sampled catalog CE with 1,024 negatives reaches
  0.047924 / 0.090300, matching the full-catalog training regime closely.
  Source: `runs/sampled_catalog/video23_half_pairwise/uniform_corrected_k1024/result.json`.
- The guarded residual-over-AR verifier selected epoch 0.  Learned residual
  epochs did not beat the original fusion; the final 0.048585 / 0.091123 is
  exactly the existing baseline, not a new gain.
  Source: `runs/verifier_co_design/video23_guarded_ar_residual_k72/result.json`.

## Terminology that is easy to confuse

- **Standalone AR:** freely/sequentially generates constrained SID paths with
  beam search.
- **AR candidate verification only:** the one-pass drafter first supplies 72
  complete legal paths; the same standalone AR checkpoint teacher-forces and
  scores only those paths.  It does not generate the candidate set.
- **Fusion:** for the same candidates,
  `score = (1-alpha) * z(drafter_score) + alpha * z(AR_score)`.
  This is score interpolation, not per-sample MoE routing.
- **Random one-pass:** random parameter initialization followed by full-data
  training.  Candidate generation is deterministic at evaluation.
- **DiffGRM direct:** iterative masked decoding/reveal policy that directly
  outputs recommendations.
- **DiffGRM-pretrained one-pass:** initializes the one-pass masked decoder from
  a DiffGRM checkpoint, then trains catalog/pairwise objectives.  Its final
  inference is still one-pass, not iterative diffusion.

## Recommended next analysis: drafter/AR complementarity

Do this before raw parameter-geometry analysis.

For every full-test example on Video23 and Music23, dump:

- target item and target codes;
- whether target enters drafter Top-10/32/72;
- target rank under drafter, AR-only reranking, and fusion;
- normalized/raw drafter and AR target scores and target margins;
- drafter unary and pairwise contributions;
- per-coordinate AR log-probabilities;
- standalone AR beam membership and rank if available;
- target popularity, train frequency, last-item-to-target transition count,
  repeat/novel flag, history length/diversity, text-kNN distance, and SID
  Hamming distance to recent history.

Primary reports:

1. Both-correct / drafter-only / AR-only / neither Hit@10 quadrants.
2. Oracle union and fusion rescue/harm rates.
3. Rank migration, especially targets initially at drafter ranks 33--128.
4. Score correlation and disagreement versus correctness.
5. The above broken down by popularity, transition frequency, repetition,
   history complexity, semantic distance, and SID geometry.

Only after the behavioral split is clear, compare hidden representations with
CKA/SVCCA, kNN overlap, effective rank, and linear probes.  Raw parameter
cosines are not reliable because hidden spaces have rotation, scaling, and
permutation symmetries.  A stronger causal test is an encoder/head cross-swap:
let the drafter catalog head consume AR history features and let the AR path
head consume drafter history features through small adapters.

## Safe first task for a new agent

Implement a read-only/full-test complementarity diagnostic that writes a new
artifact under `runs/complementarity_analysis/`.  It should reuse existing
checkpoints, not retrain or alter them, and should first reproduce aggregate
metrics from the source `result.json`.  Begin with Video23, then run Music23.

Before changing code, report:

1. which checkpoints and configs will be loaded;
2. exact candidate K and fusion alpha;
3. how concrete item identity and collision-free evaluation are preserved;
4. what per-example fields will be saved;
5. estimated memory/runtime.
