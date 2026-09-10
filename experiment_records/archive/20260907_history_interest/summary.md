# History-interest suite: reviewed conclusion

The four-arm Science validation screen and the predeclared Video transfer/test
suite completed. Two-interest attention produces only tiny Top-10 test changes
over the pooled fused-checkpoint control: +0.000134 NDCG / +0.000275 Recall on
Science and +0.000022 / +0.000127 on Video. Every paired 95% interval crosses
zero, while NDCG@5 and Recall@5 decline.

The two branch Top-10 Jaccards are 0.980660 (Science) and 0.978575 (Video).
Non-collapsed gate weights therefore do not amount to distinct-interest
retrieval. Do not promote this arm to the main method or automatically scale it
to four/eight interests.

Fused-NDCG checkpoint selection improves Video L20 over candidate-recall
selection but slightly hurts Science, so it remains a useful control rather
than a prospectively established universal rule. Local Latte remains ahead on
Video L20. See `metrics.json` for the full normalized table.

The local source of record is
`runs/history_interest_20260907/v1/FINAL_FINDINGS.md`, including paired
bootstrap intervals and artifact-verification details.
