# Latte-aligned comparison

The primary first run is Amazon23 Industrial and Scientific because Latte uses
exactly 8 latent tokens there and its full-data target is relatively quick to
evaluate.  All methods use the same 5-core catalog, sliding training examples,
leave-one-out targets, L20 history, and Recall/NDCG at 5 and 10.

Latte paper targets (Sentence-T5-base, RQ-KMeans 3x256, PSID/ESM, 8 latent,
max aggregation, final beam 500):

| Method | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| PSID | 0.0289 | 0.0445 | 0.0185 | 0.0235 |
| Latte | 0.0304 | 0.0470 | 0.0196 | 0.0249 |

Generate the matched text embeddings and SIDs, then train our AR and one-pass
pairwise-drafter + latent-route AR system:

```bash
python scripts/encode_latte_metadata.py ...
python scripts/generate_latte_rqkmeans_sids.py ...
bash experiments/latte_aligned/run_science23_latte_aligned.sh 0 all
```

The reported Latte number is only directly comparable if the SID diagnostics
confirm 25,848 unique final tuples and the run uses the exact config above.

Video Games is queued as the second independent, 8-latent confirmation.  Its
reported Latte target is Recall@5/10 = 0.0618/0.0958 and NDCG@5/10 =
0.0406/0.0515.
