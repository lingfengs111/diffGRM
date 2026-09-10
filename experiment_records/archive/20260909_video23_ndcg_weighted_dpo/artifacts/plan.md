# Rank-aware AR verifier objectives (2026-09-09)

## Question

Can the canonical Video23 OPQ4 AR verifier improve top-10 ranking when its
adaptation objective reflects SID-prefix survival, its own current errors, or
reference-relative Delta-NDCG, rather than only full-path listwise CE on fixed
drafter-rank negatives?

All arms start from the same canonical AR and frozen one-pass pairwise drafter.
Candidate budget remains K=72, validation selects the checkpoint and fusion
alpha, and the selected checkpoint is evaluated once on the complete test set.
Epoch zero remains eligible, so an arm that never improves validation reports
the untouched baseline instead of a degraded final epoch.

## Arms

| Arm | Additional objective | Training examples | Negatives |
|---|---|---|---|
| `apao_all` | 0.1 x adaptive prefix sampled-softmax | all | drafter rank strata 8/4/3 |
| `apao_support` | same | target truly in drafter Top-72 | drafter rank strata 8/4/3 |
| `ar_hard_support` | 0.05 x full-path listwise CE | target truly in Top-72 | current AR's hardest 15 mistakes from Top-72 |
| `lambda_dpo_support` | 0.1 x reference-relative pairwise DPO, weighted by Delta-NDCG@10 | target truly in Top-72 | drafter rank strata 8/4/3 |

Every arm retains token CE. The APAO and AR-hard arms also retain a frozen
teacher path-distribution KL anchor. Only the final AR decoder block, final
normalization, and BOS embedding are trainable. Prefix duplicates are removed
at each depth; APAO sampled negatives are scaled to the number of distinct
legal catalog prefixes, following the authors' official pairwise code.

`apao_all` versus `apao_support` tests whether prefix supervision should improve
the AR globally or only adapt it on verifier-reachable examples. The other two
arms isolate negative-distribution and preference-objective changes.

## Gates

1. CPU loss/unit checks must pass.
2. Each arm must finish a one-epoch 512-train/256-validation/256-test GPU smoke.
3. Full runs use six epochs, validation patience two, and a minimum of two
   adaptation epochs.
4. A result is promising only if validation NDCG@10 beats epoch zero without
   losing Recall@10. Test results are confirmatory and are not used for model
   selection.

The suite records independent logs and artifacts under
`runs/verifier_objectives_20260909/`; no existing checkpoint or result is
overwritten.

## Lambda-DPO follow-up

The local `lambda_dpo_support` label is shorthand for a LiPO-lambda-inspired,
NDCG-weighted DPO adaptation; it is not a verbatim implementation or the name
of a single source method. It combines the reference-relative log-ratio from
DPO with LambdaRank/LambdaLoss-style Delta-NDCG pair weights. Our adaptation
uses one observed next item against 15 drafter-stratified negatives, computes
weights from the fused drafter/verifier ranking, gates on target membership in
Top-72, and retains token CE.

Because the pilot's best validation checkpoint was the final allowed epoch,
`launch_lambda_repeats.sh` runs seeds 2026--2029 in parallel for at most 12
epochs. Patience is four and the minimum run length is eight epochs. This
separates a potentially truncated pilot from seed sensitivity; hyperparameters
remain unchanged.
