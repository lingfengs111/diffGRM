# Objective-ablation negative results

The tested Domino-style causal head improves standalone drafter validation
ordering, but its evidence is redundant with the independent AR verifier and
validation selects `beta=0` for the final pipeline.

The subsequent full AR-residual-distillation + candidate-listwise objective
changes test NDCG@10 by only +0.000003988, while test Recall@10 drops
0.000137186 and candidate Recall@72 drops 0.000358794. A boundary-negative
medium diagnostic recovers a little proposal coverage but lowers final
NDCG@10. The standalone ranks-11--32 promotion loss is also rejected by its
predeclared validation guard, selecting its epoch-0/beta-0 baseline.

Do not scale these exact objectives. The results distinguish candidate
coverage at K=72 from useful Top-10 ordering and show that simply imitating the
AR residual can weaken the specialization that makes fusion work.
