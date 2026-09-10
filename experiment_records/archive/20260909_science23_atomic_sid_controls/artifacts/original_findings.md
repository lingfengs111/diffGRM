# Atomic item table versus semantic-ID controls

Completed on 2026-09-09 on the full Science23 L20 split (50,985 test
examples).  In the three matched arms, the frozen history encoder/checkpoint,
trainable pooling trunk, optimizer budget, full-catalog CE, data order,
candidate budgets, AR verifier, and validation-selected fusion protocol are
identical.  Only the catalog parameterization changes.

## Matched frozen-encoder comparison

| Catalog scorer | Trainable drafter params | K | Candidate Recall | Candidate MRR | Fusion NDCG@10 | Fusion Recall@10 | Selected alpha | End-to-end ms/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Atomic item table | 6,815,233 | 72 | 0.127998 | 0.018725 | 0.022730 | 0.043778 | 0.50 | 0.613 |
| OPQ unary | 723,457 | 72 | 0.074159 | 0.013050 | 0.020308 | 0.038364 | 0.90 | 0.642 |
| OPQ pairwise | 887,490 | 72 | 0.116878 | 0.017816 | 0.022005 | 0.041797 | 0.75 | 0.779 |
| Atomic item table | 6,815,233 | 128 | 0.168599 | 0.019149 | 0.022740 | 0.043797 | 0.50 | 0.804 |
| OPQ unary | 723,457 | 128 | 0.100481 | 0.013322 | 0.020868 | 0.039757 | 0.90 | 0.793 |
| OPQ pairwise | 887,490 | 128 | 0.154967 | 0.018212 | 0.021983 | 0.041816 | 0.75 | 0.850 |

Parameter breakdown:

- atomic item table: 6,617,088 parameters;
- four OPQ code tables: 262,144 parameters;
- common history/query trunk: 198,145 parameters;
- four semantic heads: 263,168 parameters;
- pairwise selector: 164,033 parameters.

## Completely SID-free system

This control uses atomic item IDs in the history encoder, catalog retriever,
candidate representation, and proposal-aware MLP ranker.  It contains no
quantizer, SID table, SID decoder, or AR path verifier.

| K | Candidate Recall | Candidate MRR | Fusion NDCG@10 | Fusion Recall@10 | Selected alpha | End-to-end ms/example |
|---:|---:|---:|---:|---:|---:|---:|
| 72 | 0.114936 | 0.015976 | 0.018290 | 0.035344 | 0.25 | 0.0255 |
| 128 | 0.156301 | 0.016405 | 0.018302 | 0.035285 | 0.25 | 0.0303 |

The atomic retriever has 7,741,185 parameters and the atomic candidate ranker
has 264,705 parameters.  Its test retriever-only NDCG@10/Recall@10 is
0.018054/0.034598; the trained ranker alone does not improve it consistently.

## Interpretation

1. Pairwise modeling is not a cosmetic addition.  At K=72 it recovers
   +0.042718 absolute candidate recall over unary (0.116878 versus 0.074159),
   while adding only 164K trainable parameters.
2. The independent atomic table gives the strongest matched candidate recall,
   but uses about 25.2 times as many catalog-table parameters as OPQ4
   (6.62M versus 0.262M).  This is a capacity/scalability trade-off, not proof
   that semantic tokenization is intrinsically harmful.
3. More candidate recall is not automatically converted into ranking quality.
   Moving from K=72 to K=128 adds roughly four percentage points of candidate
   recall in every arm, while final NDCG@10 is flat or slightly worse.
4. Removing SID and the AR path verifier entirely causes a large final-quality
   loss.  At K=72, the no-SID system reaches 0.018290 NDCG@10 versus 0.022730
   for atomic proposals plus the frozen AR verifier.  The AR/path model is
   therefore doing more than generic candidate reranking.
5. The established production OPQ4 one-pass pairwise + AR run remains stronger
   (0.024303 NDCG@10 / 0.045386 Recall@10 at K=72) than every new matched arm.
   Its advantage can come from its independently trained history encoder and
   catalog-plus-token objective; the matched study intentionally removes those
   factors to isolate catalog representation.

Latency values are directly comparable within the three matched arms.  The
SID-free system uses a different batching/model path, so its much smaller
latency should not be treated as a hardware-identical latency comparison.
