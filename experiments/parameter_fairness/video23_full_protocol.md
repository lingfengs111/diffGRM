# Full Video23 parameter-fair verifier controls

All controls use the canonical collision-free Video23 protocol: 530,300 train,
94,762 validation, 94,762 test examples, 25,612 concrete items, history length
50, and proposal K=72.  The frozen proposal checkpoint is
`runs/diffusion_necessity/video23_full_20260829/diff_pretrained_pairwise_r51/best.pt`.

1. `light_distilled`: 627,585-parameter bidirectional path verifier, trained
   from the same random seed as the non-distilled control.  It adds KL
   distillation from the frozen standalone AR scores to label CE and margin
   loss.  The AR teacher is training-only.
2. `independent_nonar_parammatch`: a separate three-layer causal SID history
   encoder plus two path-mixing and two candidate-set layers.  The verifier
   side has 3,209,089 parameters versus 3,235,840 for the full AR verifier
   (0.83% difference).  It is trained from scratch without AR distillation.
3. `shared_encoder_ar`: the one-pass drafter history states feed the AR decoder
   directly.  The redundant AR item MLP, position table, and encoder blocks are
   bypassed.  The standalone AR decoder is first evaluated without adaptation,
   then adapted by the canonical four-coordinate teacher-forced token CE.
4. `sasrec_long_parammatch_6p7m`: canonical full-softmax SASRec with hidden
   width 216 and 6,694,613 parameters, within 0.6% of the 6,733,875-parameter
   full hybrid.  All non-capacity settings match the published 128-wide run.

Validation selects checkpoint and score-fusion alpha; test is evaluated once.

