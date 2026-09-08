# Canonical baselines and complementarity

The retained core result is a collision-free, one-pass structured semantic
retriever plus an independently trained causal AR path verifier. The Video23
canonical table is historical L50; Music23 is L20. These rows must not be
silently pooled with the newer Video23 L20 comparison.

On Video23 L50 the DiffGRM-initialized one-pass pairwise + AR system reaches
0.048585 NDCG@10 / 0.091123 Recall@10. On Music23 L20, random initialization
is slightly better than DiffGRM initialization and reaches 0.032250 / 0.059855.
This supports the structured-drafter/AR factorization but not a requirement for
diffusion pretraining.

The complementarity audit finds materially different drafter-only and AR-only
hit sets, a 0.114624 Video and 0.074235 Music oracle-union Recall@10, and strong
gains from pairwise tuple structure over unary coordinate heads. The causal AR
is weak at open-ended first-token retrieval but useful at scoring supplied
complete paths. Soft fusion beats learned hard routing in the tested setup.

See `RESEARCH_HANDOFF_2026-09-01.md` and the local
`runs/complementarity_analysis/COMPLEMENTARITY_FINDINGS_2026-09-02.md` for the
full protocol, confidence intervals, and mechanism diagnostics.
