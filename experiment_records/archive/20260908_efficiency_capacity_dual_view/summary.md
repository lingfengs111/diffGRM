# Efficiency, capacity, and dual-view controls

All rows are full Amazon23 L20 test evaluations, seed 2026, K=72,
collision-free concrete-item scoring, with checkpoint/fusion choices made on
validation.

The importance-corrected sampled-catalog objectives with 256, 1,024, or 4,096
negatives make full-catalog training cheaper, but none improves over the
Science23 full-catalog reference (0.024303 NDCG@10 / 0.045386 Recall@10).
They are useful engineering options, not main-line quality gains.

The initial Science23 capacity reductions also fail to improve NDCG. The
d176 arm preserves the result best; aggressive d128 compression is clearly
harmful. A later, more targeted drafter/verifier reallocation is recorded in
the September 9 controls.

Cross-tokenizer verification is domain-dependent. OPQ4 proposals verified by
an RQ-KMeans3 AR model improve the Video23 candidate-selected OPQ4 baseline
(0.047899 / 0.089793) to 0.048729 / 0.092126. On Music23, the matched RQ3 row
has the best Recall@10, while both RQ3 rows are essentially tied on NDCG@10.
This supports complementary views as a real phenomenon, but not a universal
fixed direction.

The Music RQ view was built from the exact pretransformed PCA representation
used by OPQ, avoiding an accidental embedding-preprocessing confound.
