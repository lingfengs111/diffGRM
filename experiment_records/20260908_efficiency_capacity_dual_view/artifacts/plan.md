# Overnight experiment queue: 2026-09-07 to 2026-09-08

All experiments use full datasets, history length 20, seed 2026, collision-free item-level evaluation, proposal K=72, and validation-only checkpoint/alpha selection.

## GPU 0: Video23 dual-view transfer

1. Train no-latent RQ-KMeans3 AR verifier.
2. Train matched RQ-KMeans3 one-pass pairwise drafter.
3. Evaluate OPQ4 drafter -> RQ-KMeans3 AR.
4. Evaluate the reverse RQ-KMeans3 drafter -> OPQ4 AR.

## GPU 1: Science23 sampled-catalog CE

Run 256, 1,024, and 4,096 uniform legal-item negatives. Training uses an importance-corrected estimate of the full-softmax denominator; validation/test always score the full 25,848-item catalog. This queue shares GPU 1 with the tail of the already-running OPQ4-Hungarian job because both have low memory occupancy.

## GPU 2: Music23 dual-view transfer

Build an RQ-KMeans3+ESM view from the exact normalized PCA-256 vectors used by the existing OPQ4 run, then run the same two-direction dual-view matrix as Video23.

## GPU 3: Science23 capacity study

1. Compact: d=128, drafter depth 2, AR depth 2+1.
2. Depth-only: d=256, drafter depth 2, AR depth 1+1.
3. Width-only: d=176, drafter depth 4, AR depth 2+2.

Each arm trains both its AR verifier and its one-pass pairwise drafter from scratch with long early-stopped schedules.
