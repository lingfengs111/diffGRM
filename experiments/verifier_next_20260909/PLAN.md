# Verifier experiments (2026-09-09)

All experiments use full Amazon23 splits with history length 20.  Test is
evaluated only from the validation-selected checkpoint and fusion weight.

## Capacity reallocation on Scientific

The frozen d176 four-layer one-pass pairwise drafter has about 1.73M
parameters.  It is paired with either a deeper d256 2+4 AR or a wider d320
2+2 AR, targeting the same roughly 6.7M total parameter budget as the
standard d256 drafter + d256 AR system.

## Proposal-aware verifier adaptation on Video Games

The canonical OPQ4 pairwise drafter and standalone AR are frozen/initialized
from their full-data checkpoints.  Candidate ranking loss is applied only
when the positive is genuinely present in the drafter top-72.  Negatives are
stratified across ranks 1-10, 11-32, and 33-72.  Original token CE and frozen
teacher path-distribution KL preserve generative competence.  The two arms
adapt either the last decoder block or all decoder blocks.

After the GPU-2 verifier run, a fixed-budget generation control runs on the
same GPU: 56 one-pass proposals plus 16 genuinely generated constrained-AR
beam proposals are deduplicated and filled back to exactly 72 candidates.
This tests whether AR generation contributes candidates unavailable to the
parallel retriever without receiving a larger final candidate budget.
