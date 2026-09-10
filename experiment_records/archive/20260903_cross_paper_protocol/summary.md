# Cross-paper comparison protocol audit

The screenshots cover different tables and dataset generations: RPG and
original DiffGRM report Amazon14 domains, while Latte reports Amazon23
Instruments, Scientific, and Games. Beauty14 is a useful RPG/DiffGRM legacy
anchor but is not a native Latte dataset. Video23 L20 is the recommended
primary common dataset; Music23 L20 is the lower-cost pilot.

The primary local comparison should rerun SASRec, original DiffGRM, RPG, PSID,
Latte, and the current structured-drafter + AR method on identical CleanGR
splits with L20, concrete-item collision handling, validation-only selection,
and shared latency/parameter reporting. Published values belong in a separate
literature-context table.

Beauty14 confirms why this separation matters. The published-style raw OPQ4
DiffGRM prediction reproduces SID-level NDCG@10 near 0.0501, but a uniform
collision-corrected concrete-item expectation is only 0.0261; collision-free
retraining is the defensible primary evaluation. RPG's much longer OPQ32 code
has negligible collision in the local artifact. Never mix L50 with L20 or
SID-bucket hits with exact-item hits in an unlabeled result table.
