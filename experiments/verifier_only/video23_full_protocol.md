# Video23 verifier-only controls

The three primary arms freeze the same collision-free OPQ4 one-pass drafter
and rank-51 pairwise selector.  They receive the same top-71 hard negatives
plus the positive item, use the same random seed and data order, and differ
only in the complete-SID path mixer:

- `bidirectional`: one bidirectional coordinate Transformer layer;
- `causal`: the identical layer with a causal coordinate mask;
- `mlp`: a parameter-matched flattened-tuple residual MLP.

All arms use last-valid-state history pooling, one candidate-set Transformer
layer, catalog-level proposal score as a feature, listwise CE plus margin loss,
and no AR distillation.  Validation selects the fusion coefficient; test is
evaluated once at that coefficient.  The frozen AR teacher is deliberately not
loaded in this phase so the comparison isolates verifier architecture and does
not spend compute on a zero-weight distillation term.

Follow-up controls, after selecting the strongest architecture, are:

1. mean versus last-valid history pooling;
2. random listwise training versus frozen-AR distillation;
3. one batched 72-candidate pass versus exact cached AR teacher forcing;
4. matched small/medium verifier capacity.
