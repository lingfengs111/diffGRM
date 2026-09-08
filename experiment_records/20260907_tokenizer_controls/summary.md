# Tokenizer/control suite: current snapshot

The strongest completed Science23 row is OPQ4 drafter proposals reranked by an
independently trained RQ-KMeans3 AR verifier: 0.019321 NDCG@5, 0.030009
Recall@5, 0.024879 NDCG@10, and 0.047230 Recall@10. The reverse dual-view does
not help. This asymmetry is compatible with complementary views, but it is not
a tokenizer-only causal comparison because both the verifier representation
and its learned model change.

All formal rows use injective, concrete item identities despite high raw SID
collision rates. ESM and Hungarian are alternative collision-repair mappings;
their rows should not be described as raw bucket-level evaluation.

The original dual-view `result.json` files are invalid because the runtime head
temperature did not match training. Only `result_v2.json` is used in
`metrics.json`. The matched OPQ4/Hungarian arm was still actively training at
this snapshot and is deliberately absent from the completed result table.
