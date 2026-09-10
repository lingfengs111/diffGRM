# Verifier and generation controls — 2026-09-09

All main comparisons below use full test sets and validation-selected fusion
weights. Random-SID drafter numbers are explicitly preliminary until its queued
run writes `result.json`.

## Video23 verifier controls

| Method | NDCG@10 | Recall@10 | Candidate Recall@72 | ms/example |
|---|---:|---:|---:|---:|
| OPQ4 one-pass pairwise + cached AR | 0.047899 | 0.089793 | 0.232984 | 1.337 |
| proposal-aware, full decoder | 0.047704 | 0.089445 | 0.232984 | 1.638 |
| proposal-aware, last decoder block | 0.047764 | 0.089909 | 0.232984 | 1.654 |
| fixed K=72: drafter-56 + true AR beam-16 | 0.047718 | 0.089857 | 0.237933 | 2.298 |

The fixed-budget generation union raises candidate recall by 0.004949 absolute,
but not final NDCG. It adds only 6.27 unique AR candidates on average, rescues
1.473% of samples, loses 0.653% previously recalled by drafter-72, and costs
about 72% more latency.

## Video23 standalone AR beam sweep

| Actual search width | NDCG@10 | Recall@10 | test wall time |
|---:|---:|---:|---:|
| 128 (canonical) | 0.043616 | 0.082712 | 8m16s |
| 256 | 0.043616 | 0.082712 | 14m39s |
| 500 (256 at digit 0) | 0.043616 | 0.082712 | 22m00s |

The top-10 list has already converged at width 128. Increasing beam only raises
cost on this tokenizer/checkpoint.

## Science23 capacity reallocation

| Drafter + AR allocation | Total params | NDCG@10 | Recall@10 | ms/example |
|---|---:|---:|---:|---:|
| d256 drafter + d256 2+2 AR | 6.719M | 0.024303 | 0.045386 | 1.416 |
| d176 drafter + d256 2+4 AR | 6.539M | 0.024180 | 0.045974 | 1.317 |
| d176 drafter + d320 2+2 AR | 6.689M | 0.023670 | 0.045602 | 1.030 |

Moving capacity from the drafter to a deeper verifier preserves NDCG and
slightly improves recall with fewer parameters. Width is less effective than
decoder depth, but neither setting beats the canonical model on NDCG.

## Matched random-SID control on Science23

| Standalone AR tokenizer | NDCG@10 | Recall@10 | full-SID top-1 |
|---|---:|---:|---:|
| semantic OPQ4 | 0.020772 | 0.039051 | 0.007453 |
| exact OPQ4 path set randomly reassigned to items | 0.015389 | 0.029165 | 0.005727 |

Random reassignment loses 25.9% NDCG and 25.3% Recall despite training longer
(best epoch 114; early stop at 154). Teacher-forced digit-0 accuracy falls from
0.04578 to 0.01569, whereas later conditional digits remain comparatively easy.
This locates most semantic value in the initial catalog routing decision.

The random-SID drafter is still running. Its best validation candidate
Recall@72 so far is 0.102913 (epoch 33), versus 0.143552 for semantic OPQ4.
