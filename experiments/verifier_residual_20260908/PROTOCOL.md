# Residual verifier screening, 2026-09-08

## Question

The direct whole-SID scalar verifier replaced a strong AR path likelihood and
lost 9.84% NDCG@10 on Video23 validation. This experiment instead asks whether
a learned whole-SID score can *correct* the frozen AR verifier while preserving
its calibrated joint likelihood.

The Video23 L20 OPQ4/ESM, K=72, proposal cache, AR checkpoint, drafter, and
validation examples are exactly the frozen artifacts from
`runs/verifier_arch_20260908/v2/full/video23/cache`. The previous cache has
already reproduced all 28 original AR-fusion metrics within 1e-9. The cache is
treated as read-only and re-hashed by every process. No test split is read,
scored, or selected on. No tokenizer, item metadata, candidate budget, or
negative sampler changes.

## Arms

| Arm | Total verifier score | Candidate self-attention |
|---|---|---|
| A | Original frozen AR path log-likelihood | Causal AR |
| B | `s_AR + r_theta(history, complete SID)` | Causal |
| C | `s_AR + r_theta(history, complete SID)` | Bidirectional |

Both B and C use the same existing AR model, history cross-attention, 66,049
parameter H→H→1 residual MLP, initial random state, data order, and hyperparameters.
The final MLP projection is zero-initialized, so `r_theta=0` exactly before
training. Its bounded form is `2*tanh(raw/2)`: a correction can reorder nearby
items but cannot replace the AR score wholesale. The observed original AR
within-query score standard deviation has median 1.25, and its score range has
median 6.01, giving the cap a concrete scale.

The AR encoder, decoder, embeddings, BOS, and norms remain frozen. Therefore
the AR score is computed once for each positive-first train candidate set and
stored in a separate checksum-verified array; validation uses the frozen AR
scores already present in the predecessor cache. There is no decoder update,
token CE, imitation, or second model. The sole train loss is listwise CE on
the actual residual total score, plus `0.01 * mean(residual^2)`.

Training candidates remain target at column 0 plus the top 71 non-target
drafter candidates. Validation remains untouched drafter Top-72 with no target
injection. Candidate scores are independent of candidate order and chunks.

## Selection and interpretation

Epoch zero is saved as a selectable checkpoint and must reproduce every
frozen-AR validation metric within 1e-8. The model then trains eight epochs,
physical batch 64 with accumulation 2, AdamW head LR 1e-4, weight decay 1e-4,
gradient clip 1.0, seed 2026. Fusion alphas are fixed at
`0,.1,.25,.5,.75,.9,1`; validation NDCG@10 selects across all alphas and all
epochs including epoch zero. Thus no residual outcome can be reported as worse
than A merely because the comparison omitted the untouched baseline.

Save ranks and rescue/harm counts. B must exceed A to support residual
correction. C must exceed B to support bidirectional complete-SID interaction.
This is one-seed validation screening; even a positive outcome requires later
seeds and a held-out test confirmation. Science, Music, hard-negative mining,
and retrieval comparisons are deliberately not launched here.

## Running

`python experiments/verifier_residual_20260908/prepare_suite.py --root RUN_ROOT`
freezes code plus hashes of the predecessor candidate cache. Then:

`bash experiments/verifier_residual_20260908/launch.sh RUN_ROOT SESSION`

The tmux supervisor first runs a small score-cache/training smoke test. It then
precomputes frozen AR scores for all 530,300 training candidate sets on GPU 2
and launches B/C on GPUs 0/1 after it completes. GPU 3 is not used. Inspect
`suite_status.json`, `supervisor.log`, per-arm `progress.json`, and
`decision.json`.
