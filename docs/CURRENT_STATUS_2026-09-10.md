# Current research status (2026-09-10)

This is the entry point for the second GitHub research snapshot. The detailed
architecture/code map remains in `RESEARCH_HANDOFF_2026-09-01.md`; the
September 7 status and older `runs/` summaries are historical. Normalized
metrics and the small original result files for every closed suite below are
tracked under `experiment_records/`.

## Current method and fixed protocol

The main system is a collision-free one-pass structured semantic retriever
plus an independently trained causal AR verifier. The drafter scores the legal
item catalog with unary coordinate scores and learned pairwise tuple
compatibility, retrieves K=72 concrete items, and the AR teacher-forces every
proposed complete SID in parallel. Validation selects the fusion alpha.

Current Amazon23 experiments use the latest 20 history items, exact concrete-
item evaluation, collision-free catalogs, and seed 2026 unless explicitly
marked multi-seed. Historical Video23 L50 numbers and current L20 numbers must
remain separate.

## Strongest current orientation rows

These are not yet a final matched multi-seed paper table.

| Dataset | Method | NDCG@10 | Recall@10 | Qualification |
|---|---|---:|---:|---|
| Video23 | local Latte, beam 500 | 0.051082 | 0.095407 | different architecture and generation budget |
| Video23 | pooled pairwise + AR, fused-checkpoint control | 0.048461 | 0.091281 | strongest current version of our base pipeline |
| Video23 | NDCG-weighted DPO | 0.048258 +/- 0.000033 | 0.090820 +/- 0.000069 | four-seed mean; improves its 0.047899/0.089793 fixed baseline |
| Music23 | RQ-KMeans3 drafter + matched AR | 0.032319 | 0.061474 | highest Recall@10 |
| Music23 | RQ-KMeans3 drafter + OPQ4 AR | 0.032320 | 0.060516 | highest NDCG@10 by a negligible margin |
| Science23 | OPQ4 drafter + RQ-KMeans3 AR | 0.024879 | 0.047230 | current best dual-view row |

The DPO result is a real and unusually consistent verifier gain, but it is not
yet the project's absolute Video23 best. Its frozen candidate-selected drafter
baseline is 0.047899/0.089793; the independently selected fused-checkpoint
control already reaches 0.048461/0.091281. The clean next test is to apply the
same fixed DPO recipe to that stronger drafter/checkpoint protocol.

## Results added after the first snapshot

### Rank-aware verifier objectives

APAO-style prefix training and current-AR hard-negative listwise training are
negative or non-promotable in the tested form. The successful arm combines a
reference-relative DPO gap with Delta-NDCG@10 pair weights, gates training on
targets actually present in drafter Top-72, samples negatives from ranks
1-10/11-32/33-72, retains token CE, and updates only the last decoder block.

Seeds 2026-2029 all improve both Top-10 metrics. The mean gains over the fixed
baseline are +0.0003594 NDCG@10 (+0.75%) and +0.0010263 Recall@10 (+1.14%),
with unchanged candidate recall. The internal `lambda_dpo_support` label is
shorthand for this local LiPO-lambda-inspired construction, not a verbatim
paper implementation.

### Architecture and residual verifier screens

On the immutable Video23 validation proposal cache, replacing AR path
likelihood by a learned complete-SID scalar head loses about 0.0054 NDCG@10.
Causal and bidirectional variants both fail. Adding a bounded, zero-initialized
whole-SID residual over the frozen AR is safer, but validation chooses epoch 0
for both variants; every learned epoch is rejected. These are validation-only
negative results. The whole-SID v1 cache had a temperature error and is invalid;
only whole-SID v2 and residual v3 are formal.

### Generation controls

A fixed 72-candidate union of drafter-56 plus true AR beam-16 increases Video23
candidate recall from 0.232984 to 0.237933, but does not improve final NDCG and
costs about 72% more latency. Standalone AR Top-10 is exactly unchanged at
actual search widths 128, 256, and 500. Wider generation is therefore not the
current bottleneck for this tokenizer/checkpoint.

### Capacity reallocation

On Science23, a smaller d176 drafter plus deeper d256 2+4 AR uses fewer total
parameters than the canonical d256+d256 2+2 system and raises Recall@10 from
0.045386 to 0.045974, while NDCG@10 slips from 0.024303 to 0.024180. Widening
the AR to d320 performs worse. Decoder depth is more useful than width here,
but neither arm replaces the NDCG-leading model.

### Semantic-ID necessity controls

The matched random-SID run has now finished. It preserves the exact legal OPQ4
path multiset but randomly reassigns paths to items. Standalone AR falls from
0.020772/0.039051 to 0.015389/0.029165, and the complete random-SID
drafter+verifier reaches only 0.091929 candidate Recall@72 and 0.015831
NDCG@10, versus 0.122212 and 0.024303 for semantic OPQ4. Much of the semantic
advantage appears in the first routing coordinate.

In the matched atomic-table study, atomic proposals have the best K=72
candidate recall (0.127998), but the item table uses about 25.2 times the
catalog parameters of OPQ code tables. OPQ pairwise scoring raises candidate
recall from 0.074159 to 0.116878 over OPQ unary with only 164K extra parameters.
The wholly SID-free retriever/ranker reaches 0.018290 NDCG@10, well below the
0.024303 established OPQ4+AR system. Semantic IDs are therefore justified by
parameter sharing, structured interactions, and the AR path model—not by the
claim that every semantic tokenizer must beat a large atomic table.

### Efficiency and cross-view controls

Importance-corrected sampled-catalog CE with 256, 1,024, or 4,096 negatives is
close to full-catalog training but does not improve it. It remains an optional
training-efficiency tool. Cross-tokenizer verification helps Video23 in the
OPQ4-to-RQ3 direction and gives a Music23 Pareto improvement, but the preferred
direction is domain-dependent.

## Stable interpretation

- The main contribution remains complementary factorization: efficient
  structured joint-tuple retrieval plus independent causal path verification.
- Candidate coverage alone is not enough. K=128, generation union, and atomic
  tables all add recall that the current ranking stage does not fully convert.
- Semantic structure matters, especially early routing, while pairwise tuple
  modeling recovers much of the gap to an expensive atomic catalog table.
- Generic whole-SID heads, bounded residuals, wider beams, and the tested
  proposal-aware CE variants are not supported by current results.
- NDCG-weighted DPO is the most credible new verifier improvement and should be
  tested next on the strongest fused-checkpoint drafter and then on Music23 and
  Science23 under an unchanged protocol.

## Evidence and checkpoint boundary

Git contains all new source/config/test files plus five new immutable evidence
records:

- `20260908_efficiency_capacity_dual_view`
- `20260908_video23_verifier_architecture_screen`
- `20260909_generation_capacity_sid_controls`
- `20260909_science23_atomic_sid_controls`
- `20260909_video23_ndcg_weighted_dpo`

Each record includes normalized metrics, a concise conclusion, and the small
original result files. Its manifest lists the local source paths and checkpoint
references. Consequently, a GitHub-only machine can understand every result
without the weights. Checkpoints, prepared data, SIDs, rank arrays, and full
logs remain outside Git and should be transferred only to the checkpoint-capable
machine. `docs/EXPERIMENT_SYNC.md` defines the ongoing multi-machine record
workflow.
