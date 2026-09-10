# Atomic item table versus semantic-ID controls

Completed on the full Science23 L20 test split (50,985 examples). In the three
matched arms, the frozen history encoder, trainable pooling trunk, optimizer
budget, full-catalog CE, data order, candidate budgets, AR verifier, and
validation-selected fusion protocol are identical. Only the catalog
parameterization changes.

At K=72, pairwise OPQ raises candidate recall from 0.074159 to 0.116878 over
unary OPQ while adding only 164K parameters. The independent atomic item table
is strongest within the matched study at 0.127998 candidate recall and 0.022730
NDCG@10, but its item table uses 6.62M parameters versus 0.262M for the four
OPQ code tables (about 25.2 times as many). This is a capacity/scalability
trade-off, not proof that semantic tokenization is harmful.

Increasing K from 72 to 128 gives roughly four points more candidate recall in
every arm without improving final NDCG@10. The ranking bottleneck remains.

The completely SID-free system reaches only 0.018290 NDCG@10 at K=72. An
atomic proposal model can work, but removing the AR path verifier loses
substantial quality. The established independently trained OPQ4 pairwise + AR
system remains stronger than every matched arm at 0.024303 NDCG@10 / 0.045386
Recall@10.

Latency is directly comparable within the three matched arms. The SID-free
implementation follows a different batching/model path, so its much smaller
latency is not a hardware-identical comparison.
