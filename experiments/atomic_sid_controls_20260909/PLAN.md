# Atomic item table versus semantic-ID controls (Science23, L20)

All matched-drafter arms use:

- the same frozen OPQ4 AR history encoder/checkpoint;
- the same trainable history pooling/query trunk and seed;
- full legal-catalog cross entropy for 16 epochs;
- the same collision-free catalog and train/validation/test examples;
- K=72 and K=128 candidate evaluation;
- the same frozen AR path verifier and validation-selected fusion weight.

The only changed factor is the catalog score parameterization:

1. `atomic`: one independent trainable vector per item;
2. `unary`: four shared code tables with additive coordinate scores;
3. `pairwise`: unary scores plus all six low-rank coordinate-pair terms.

Each result records candidate Recall/MRR/NDCG, final verifier/fusion metrics,
CUDA-synchronized latency, and a parameter breakdown.  A separate no-SID
atomic retriever/ranker control is launched by `run_atomic_no_sid.sh`.
