# Rank-aware AR verifier objectives

All arms start from the same Video23 L20 canonical AR and frozen K=72 one-pass
pairwise drafter. Checkpoints and fusion alpha are selected on validation; the
full test set is evaluated once. Epoch 0 remains selectable.

The APAO-style prefix losses and current-AR hard-negative listwise objective do
not improve the baseline. The positive result is the local NDCG-weighted DPO
arm: a reference-relative positive-versus-negative DPO loss weighted by the
current fused ranking's Delta-NDCG@10. It trains only on cases whose target is
actually inside the drafter Top-72, uses 15 drafter-rank-stratified negatives,
retains token CE, and updates the last decoder block.

The six-epoch pilot reached 0.048213 NDCG@10 / 0.090870 Recall@10, with its best
checkpoint at the final permitted epoch. The predeclared follow-up therefore
kept the hyperparameters fixed, extended training to at most 12 epochs, and
ran seeds 2026-2029.

The four-seed test mean is 0.0482580 +/- 0.0000331 NDCG@10 and 0.0908196 +/-
0.0000686 Recall@10 (sample standard deviation). Relative to the untouched
0.0478986 / 0.0897934 baseline, the mean gains are +0.0003594 (+0.75%) and
+0.0010263 (+1.14%). Every seed improves both Top-10 metrics. Candidate recall
is unchanged, so this is a verifier/ranking gain rather than additional
retrieval coverage.

The internal `lambda_dpo_support` run label is shorthand. This is a
LiPO-lambda-inspired combination of DPO and LambdaRank/LambdaLoss-style
Delta-NDCG weighting, not a verbatim implementation or the canonical name of
a single paper method.
