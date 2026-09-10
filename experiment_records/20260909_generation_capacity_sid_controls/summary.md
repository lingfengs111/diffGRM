# Generation, capacity-reallocation, and SID-semantic controls

The Video23 proposal-aware AR adaptations did not improve NDCG@10. Limiting
updates to the final decoder block preserves Recall slightly better than
adapting the full decoder, but remains below the candidate-selected baseline.

Adding true AR generation under a fixed 72-candidate budget increases coverage
from 0.232984 to 0.237933, yet final NDCG@10 falls slightly and latency rises
about 72%. It contributes only 6.27 unique candidates per example on average,
rescues 1.473% of examples, and loses 0.653% of drafter-72 positives. Separately,
standalone AR Top-10 is identical at beams 128, 256, and 500, so the search has
already converged by width 128 for this checkpoint/tokenizer.

On Science23, reallocating capacity from a d256 drafter into a deeper verifier
preserves NDCG and modestly raises Recall with fewer total parameters. A wider
verifier is less effective. Neither replaces the canonical NDCG-leading row.

The matched random-SID experiment now fully completed. Randomly reassigning the
exact collision-free OPQ4 path set to items sharply hurts standalone AR and the
trained one-pass drafter. The complete random-SID system reaches only 0.091929
candidate Recall@72 and 0.015831 NDCG@10, versus 0.122212 and 0.024303 for the
semantic OPQ4 reference. This is direct evidence that the SID semantics matter,
especially for the first routing token; it is not merely a path-capacity effect.
