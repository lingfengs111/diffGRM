# Whole-SID and residual verifier screening

This is a Video23 L20 validation-only architecture screen on an immutable K=72
proposal cache. No test data were used. The original AR path score reproduced
the predecessor metrics before either study was allowed to proceed.

Replacing the AR path likelihood with a learned complete-SID scalar head is a
clear failure: both causal and bidirectional variants lose roughly 0.0054
NDCG@10 relative to the frozen AR baseline. Bidirectional interaction does not
repair that loss.

A safer zero-initialized, bounded residual on top of the frozen AR was then
tested. In both causal and bidirectional arms, validation selected epoch 0;
all learned corrections were rejected. This means the reported equality to
the baseline is a valid negative result, not evidence that training improved
the verifier.

The v1 whole-SID cache used the wrong head temperature (1.0 instead of 0.07)
and was rejected by baseline parity. Only `v2/full` is formal. The residual
study's formal results are `v3/full`.
