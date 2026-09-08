# History attention, multiple interests, and final-metric checkpoint selection

## Fixed protocol

The suite root is `runs/history_interest_20260907/v1/`. The supervisor runs
from a source snapshot under that root. `source_manifest.json` records source,
config, AR checkpoint and SID SHA-256 hashes. Original experimental artifacts
are never overwritten. The `DIFF_GRM` package name is historical.

Science23 and Video23 use the existing pure-comparison OPQ4/ESM mappings,
Latte metadata, PCA-192, L20, full training and validation splits, seed 2026,
and K=72. All arms share the frozen domain-specific independent AR checkpoint.
Train settings match the retained pure line: encoder depth 4, pair rank 51,
batch 256, backbone LR 0.0003, selector LR 0.001, weight decay 0.0001,
catalog CE + 0.1 token CE, at most 80 epochs, patience 14, minimum 16 epochs.
No new negative-sampling scheme or tokenizer is introduced.

## Four Science arms

| Arm | History use | Interests | Selection |
|---|---|---:|---|
| pooled | original last valid encoder state | 1 | final fused validation NDCG@10 |
| mlp | extra H -> 2H -> H MLP on coordinate states | 1 | same |
| attention | each coordinate query attends to all valid history states | 1 | same |
| interest2 | two sets of coordinate queries, shared attention and pairwise factors | 2 | same |

The MLP control adds 4H²+3H reader parameters, versus attention's 4H²+4H;
both add the same normalization. A four-interest variant is implemented and
smoke-tested, but is not another full-training sweep in this initial suite.

All new heads inherit the same causal history encoder. The attention reader
uses only observed history, with padding masked. No target enters the reader.
Each interest scores the entire legal catalog with unary + pairwise terms.
Only then are complete-item energies aggregated:

`s(h,i) = T logsumexp_m(log pi_m(h) + s_m(h,i)/T)`, with T=1.

The mixture weights are a learned softmax over the pooled history; the gate
starts uniform. This is a smooth mixture of energies, not a mixture of
normalized catalog probabilities. Auxiliary token CE uses the weighted
mixture of per-interest coordinate probabilities. Pair factors are shared.
There is no distillation, hard routing, branch-specific quota, or extra AR
pass. The interest diagnostic reports gate entropy/weights, overlap of
branch Top-10 sets, and branch attribution in the aggregated Top-10, to detect
collapse. One history encoder forward is used; downstream scoring cost grows
with the number of interests and must be reported separately.

## Checkpoint selection and test discipline

Every epoch evaluates the frozen AR on the same candidate set and the existing
alpha grid `0,0.1,0.25,0.5,0.75,0.9,1`. `best.pt` maximizes fused validation
NDCG@10. `candidate_best.pt` independently tracks highest drafter Recall@72
within the same training trajectory. Its epoch's fused metrics are preserved
in `candidate_selection_validation`; both pooled checkpoints later receive
test evaluation. This isolates checkpoint selection within a common
trajectory. Early stopping is based on fusion, so it is not an exact replay
of the historical recall-based training horizon.

Full Science screening does **not** evaluate test. Among attention and
interest2, a variant qualifies for Video when validation NDCG@10 is at least
pooled + 0.0001 and validation Recall@10 does not decrease. This is a practical
screening gate, not a significance test. The qualifying variant with highest
NDCG transfers alongside pooled and the MLP control. If none qualifies,
only pooled transfers, testing checkpoint selection on Video. Decisions are
written to `transfer_decision.json` before any test evaluation. The capacity
control is reported even if it wins; attention cannot be claimed as the
cause of improvement without outperforming that control.

After Video training, the fixed baseline/control/selected-method rows are
evaluated on full Science and Video test sets with validation-selected alpha.
Per-example target SID and drafter/AR/fused ranks are saved as NPZ, in split
order. Missing targets have rank K+1; SIDs are injective concrete identities.
The suite is a single-seed screening experiment, not a final significance or
three-domain claim. Music remains a subsequent transfer once these results
are reviewed.

## Running and monitoring

`bash experiments/history_interest_20260907/launch.sh smoke` runs five short
GPU trainings and full-validation parity against the retained Science
checkpoint. Parity checks all common ranking metrics to 1e-6 absolute.

`bash experiments/history_interest_20260907/launch.sh full` requires the
successful parity artifact and detaches the supervisor. Science initially
occupies GPU 0/1/2/3 for pooled/mlp/attention/interest2 respectively.

Inspect `suite_status.json`, `full_supervisor.log`, each arm's `train.log`,
`progress.json` (updated atomically each epoch), and eventual `result.json`.
There is no automatic overwrite/restart of an incomplete arm. A failed wave
blocks subsequent dependent stages. `command.json` records exact commands.
Tree-shape diagnostics are skipped to avoid their factorial overhead; they
do not participate in scores or ranking. Reported evaluation time includes
metrics and the alpha sweep and is not a deployment-latency benchmark.

## Checks

CPU checks cover all new heads, history padding invariance, direct access to
unpooled history, finite gradients through encoder/reader/gate/selector,
mixture order/permutation invariance, old pooled strict checkpoint reload,
fusion selection and target-rank encoding. Existing structured-selector tests
remain applicable. GPU smoke uses two batches at the actual batch size 256,
the full item catalog, checkpoint reload, frozen AR and validation rank dumps.
