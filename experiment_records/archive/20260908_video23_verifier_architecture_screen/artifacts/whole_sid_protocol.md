# Whole-SID verifier screening, 2026-09-08

## Question and fixed inputs

Does a learned complete-item readout improve the existing verifier, and does
bidirectional interaction within the four OPQ coordinates improve it further?
This is a single-seed diagnostic, not a three-domain or significance claim.

Use the September 7 pooled, fused-validation-selected checkpoints on Video23
and Science23. Both domains retain L20, collision-free OPQ4/ESM, codebooks of
256 entries, and the exact same K=72 concrete-item proposals within each domain.
Video: 530,300 train / 94,762 validation / 25,612 items. Science: 259,992 train /
50,985 validation; catalog size is checked against its domain configuration.
The original standalone AR checkpoint supplies initialization and baseline A.
No tokenizer, drafter, metadata representation, or exposure label changes.

The source snapshot and copies of checkpoints, SIDs, reference results, and
configs live under `runs/verifier_arch_20260908/v2/`. Hashes pin them before
launch. The prepared raw data stay in their existing directories; their
train/validation split and item-vocabulary hashes are recorded in each cache.
Non-parameter drafter settings are restored from checkpoint `args`, including
the original logit temperature 0.07 (also applied to non-normalized logits).
The v1 cache-preparation attempt incorrectly used 1.0. Full baseline parity
rejected it before formal verifier training; its artifacts are retained only
for debugging and must not be used for scientific comparisons.

## Arms

| Arm | Candidate readout | Candidate attention | Optimization |
|---|---|---|---|
| A | Original four-token AR log-likelihood sum | Causal | Frozen baseline |
| B | Final complete-SID state -> H/H/1 GELU MLP | Causal | Ranking + causal auxiliary CE |
| C | Identical state position and MLP | Bidirectional | Identical to B |

B and C load identical AR parameters and initialize their new MLP identically.
Both consume `[BOS,c0,c1,c2,c3]`; the readout is the state at c3. Coordinate
identity is retained through disjoint embedding slices. The only architecture
difference is the candidate self-attention mask. Every decoder layer already
cross-attends to the complete history token sequence. Candidates do not attend
to one another, so reordering or chunking them must not change their scores.

Keep embeddings, BOS, history encoder, and the shared final LayerNorm frozen.
Epoch 1 trains only the new head. Epochs 2-8 adapt all decoder blocks and the
head. Frozen history computation stays in evaluation mode. Gradients must reach
decoder cross-attention Q/K/V despite the frozen history encoder.

The AR auxiliary uses a SEPARATE CAUSAL decoder pass on the positive item,
with the same trained decoder weights. Never supervise token prediction from
the bidirectional ranking pass, which can already see those tokens. Use mean
unsmoothed token CE, weight 0.1, only during decoder adaptation. This is a
regularizer, not AR distillation and not a second verifier in inference.

## Candidates and optimization

Prepare the drafter outputs once and share immutable caches between B/C.
Training: positive in column 0 plus the top 71 non-target legal catalog items.
Validation: untouched top 72, with no positive injection. The positive-column
convention cannot leak through candidate-set position: each candidate is
scored independently. An explicit permutation test checks this property.

Optimize raw verifier-score listwise cross-entropy, plus the causal auxiliary.
Do not train on a fusion of drafter/AR/rank-head scores. Inference replaces the
old verifier score with the new score; fusion still uses only drafter+verifier.
Head LR 3e-4, decoder LR 1e-5, AdamW weight decay 1e-4, gradient clipping 1.0.
Physical batch 32, accumulation 4 (effective batch 128), evaluation batch 64,
candidate chunk 16, FP32. All eight epochs run in both arms; no unequal early
stopping. Seed 2026 controls initialization; a separate deterministic epoch
permutation gives both arms identical data order. Final partial accumulation
groups are weighted by actual example count.

## Selection, interpretation, and stopping

Baseline A must reproduce all 28 existing fused validation metrics (four
metrics, seven alpha values) within absolute 1e-6 before full training.
Alpha grid: 0, .1, .25, .5, .75, .9, 1. Each arm's checkpoint and alpha maximize
validation NDCG@10; report verifier-only and fused NDCG/Recall at both 5 and 10.
Reload the selected checkpoint and verify metric parity before marking done.

Video runs first. If either B or C beats A's fused validation NDCG@10 by at
least 0.0002 without reducing Recall@10, transfer BOTH B and C to Science.
This practical screening gate is not a statistical test. Science preparation
may run alongside Video preparation to use idle resources, but Science
training depends on the Video gate. A failed smoke/parity/job blocks dependent
stages. Do not silently replace a failed/new head by original AR and call it a
new result. B must beat A to support readout adaptation; C must beat B to
support bidirectional interaction, with additional seeds needed for confidence.

Save per-example drafter, verifier-only, and selected-fusion ranks. Report
rescue/harm counts and target migration from drafter ranks 11-32 and 33-72.
Bootstrap intervals on these validation-selected checkpoints are descriptive,
conditional on one training seed; they are not independent confirmation.

This first suite does not score test or use it for selection. The inherited
dataset adapter loads all prepared split files for metadata bookkeeping, but
only train/validation examples are tokenized, cached, scored, or optimized.
Music and multi-seed/test confirmation follow review of this screening result.
Hard-negative mining and matched retrieval controls remain subsequent stages;
they are not mixed into the first architecture experiment.

## Running

`python experiments/verifier_arch_20260908/prepare_suite.py --root RUN_ROOT`
creates a new source/input snapshot and refuses overwrite. Then:

`bash experiments/verifier_arch_20260908/launch.sh RUN_ROOT SESSION_NAME`

The tmux supervisor uses GPUs 0 and 1 for matched arms, GPU 2 for the independent
Science cache preparation, and leaves GPU 3 to the existing task. Smoke checks
are gates inside the suite. Inspect `suite_status.json`, `supervisor.log`, and
each arm's `progress.json`/`result.json`. The supervisor owns only its children.

For a machine with the same environment and prepared-data paths, transfer the
new RUN_ROOT and invoke its snapshotted launch script; no modifications to the
remote live project are needed. `PYTHON_BIN` can select an equivalent conda
environment. The snapshot has checkpoint/SID copies and records data paths.
