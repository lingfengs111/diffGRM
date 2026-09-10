# Current research status (2026-09-07)

> Superseded by `docs/CURRENT_STATUS_2026-09-10.md`. This file is retained as
> the first GitHub snapshot's historical status.

This is the short current entry point. The detailed architectural handoff is
`RESEARCH_HANDOFF_2026-09-01.md`; the older
`runs/canonical_full/progress_summary_2026-08-23.md` is historical and predates
the current one-pass formulation.

## Current method and evaluation contract

The main system is no longer best described as diffusion. It uses a
collision-free OPQ semantic-ID catalog, a one-forward structured drafter with
unary and pairwise tuple scores, and an independently trained causal AR model
that teacher-forces and reranks proposed complete item paths. Validation selects
the drafter/AR fusion alpha. Formal evaluation resolves every SID to a concrete
item and reports NDCG/Recall at both 5 and 10.

Two protocol families must remain visibly separate:

- the established Video23 table in the September 1 handoff uses historical
  `L50` and reaches 0.048585 NDCG@10 / 0.091123 Recall@10;
- the September 7 Science23/Video23 suites use aligned `L20`, K=72, seed 2026,
  and validation-selected alpha 0.75.

## September 7 history-interest result

On the L20 test sets, two-interest attention does not establish a useful
improvement over the capacity-matched controls.

| Dataset | Method | NDCG@5 | Recall@5 | NDCG@10 | Recall@10 |
|---|---|---:|---:|---:|---:|
| Science23 | pooled, fused-selection checkpoint | 0.018783 | 0.029009 | 0.023881 | 0.044896 |
| Science23 | two interests | 0.018582 | 0.028263 | 0.024015 | 0.045170 |
| Video23 | pooled, fused-selection checkpoint | 0.037772 | 0.057987 | 0.048461 | 0.091281 |
| Video23 | two interests | 0.037623 | 0.057629 | 0.048483 | 0.091408 |

The two-interest Top-10 deltas are only +0.000134/+0.000275 on Science and
+0.000022/+0.000127 on Video for NDCG@10/Recall@10; paired 95% intervals cross
zero. NDCG@5 and Recall@5 decrease. Branch Top-10 Jaccard remains about 0.98,
so the branches have not learned meaningfully distinct retrieval sets. Local
Latte remains clearly ahead on Video L20 (0.051082 / 0.095407 at 10).

Checkpoint selection on fused validation NDCG helps Video L20 but not Science:
it improves Video over the candidate-recall-selected pooled checkpoint, while
slightly hurting Science. It is therefore retained as a control, not a
universal gain. Full evidence is in
`runs/history_interest_20260907/v1/FINAL_FINDINGS.md` locally and in the tracked
history-interest experiment record.

## September 7 tokenizer controls

On Science23 L20, the best completed row is currently the true dual-view setup:
OPQ4 drafter proposals verified by an RQ-KMeans3 AR model, both with
collision-free ESM mappings. It reaches 0.019321 NDCG@5, 0.030009 Recall@5,
0.024879 NDCG@10, and 0.047230 Recall@10. This is stronger than the matched
OPQ4/ESM row (0.019138 / 0.029381 / 0.024303 / 0.045386), but it changes both
the verifier view and model, so it is evidence for cross-tokenizer
complementarity rather than a clean tokenizer-only causal claim.

Raw SID collision rates vary substantially (OPQ4 32.64%, RQ2+OPQ2 15.88%,
RQ-KMeans3 31.68%, RQ-KMeans4 14.50%), but every formal row uses an injective
repaired catalog. The first two dual-view `result.json` files are invalid due
to a runtime head-temperature mismatch; only their `result_v2.json` reruns are
valid.

The matched OPQ4 + Hungarian control subsequently completed at 0.019125
NDCG@5, 0.029675 Recall@5, 0.024520 NDCG@10, 0.046445 Recall@10, and 0.125135
candidate Recall@72. Relative to matched OPQ4/ESM it improves candidate
coverage and the Top-10 metrics, but remains below the OPQ4-to-RQ-KMeans3
dual-view result. Collision repair is therefore consequential, but it does not
explain away the asymmetric dual-view gain.

## Recent objective ablations

Three plausible causal/rank-aware corrections have already received negative
or non-promotable results on the historical Video23 L50 line:

- a Domino-style positive-NLL causal residual improves the drafter's
  validation Top-10 ordering, but validation chooses `beta=0` after AR fusion;
- AR-residual distillation plus candidate-listwise hard negatives changes full
  test NDCG@10 by only +0.000004, while Recall@10 and candidate Recall@72 fall;
- a standalone ranks-11--32-to-Top-10 boundary loss is rejected by its
  predeclared validation guard on the medium diagnostic.

These results reject the exact tested objectives, not rank-aware supervision in
general. If revisited, a rank-boundary term needs a controlled weak
listwise/AR auxiliary ablation rather than another scale-up of the standalone
loss.

## Stable conclusions and next comparison work

- The most robust contribution remains the complementary factorization:
  structured one-pass joint-tuple retrieval plus independent causal path
  verification and soft fusion.
- Increasing candidate K from 72 to 128/256 barely moves Top-10 quality even
  though oracle candidate coverage rises. Ranking/verification, rather than the
  K=72 ceiling alone, is the bottleneck.
- Naive residual imitation and multiple-interest capacity have not produced a
  stable main-line gain.
- Cross-paper comparison should first freeze a shared dataset/protocol table:
  exact Amazon version/domain, core filtering, history length, item-identity
  collision handling, split, candidate budget, beam width, and checkpoint
  selection rule. Results with L50 and L20, or bucket-level and item-level
  correctness, must not share an unlabeled table row.

The tracked normalized tables are in `experiment_records/INDEX.md`. Large raw
artifacts and checkpoints remain under ignored local directories.
