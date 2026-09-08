# Amazon23 Music, Science, and Office

These domains use the same prepared-data and collision-free OPQ protocol as
Video23.  The dataset class is `AmazonReviews2023CleanGR`; the category and
prepared directory come from the domain YAML.

All new runs default to `max_history_len=20` for Latte-compatible comparison.
The existing prepared files retain histories up to 50, so no rebuild is needed:
the tokenizer selects the most recent 20. See
[`../AMAZON_MAXLEN_PROTOCOL.md`](../AMAZON_MAXLEN_PROTOCOL.md).

The canonical core5 data is already prepared and has passed the DiffGRM data
adapter audit:

| domain | interactions | users | items | train | validation | test |
|---|---:|---:|---:|---:|---:|---:|
| video | 814,586 | 94,762 | 25,612 | 530,300 | 94,762 | 94,762 |
| music | 511,836 | 57,439 | 24,587 | 339,519 | 57,439 | 57,439 |
| science | 412,947 | 50,985 | 25,848 | 259,992 | 50,985 | 50,985 |
| office | 1,800,878 | 223,308 | 77,551 | 1,130,954 | 223,308 | 223,308 |

## 1. Rebuild core5 data when needed

From the CleanGR repository:

```bash
bash scripts/prepare_amazon23_diffgrm.sh all prepare
```

Optional alternate text views do not overwrite the canonical splits:

```bash
bash scripts/prepare_amazon23_diffgrm.sh music rich_meta
bash scripts/prepare_amazon23_diffgrm.sh music rich_reviews
```

`rich_reviews` reads the deduplicated core review file and restricts snippets
to user-item pairs visible in `splits/train.jsonl`; validation/test reviews are
excluded from the item representation.

## 2. Audit prepared rows

```bash
python scripts/validate_prepared_protocol.py \
  --dataset AmazonReviews2023CleanGR \
  --config experiments/amazon23_domains/common.yaml \
  --config experiments/amazon23_domains/music23.yaml \
  --data-only
```

Replace `music23.yaml` with `science23.yaml` or `office23.yaml` as needed.
Omit `--data-only` to also build and audit the sentence-embedding OPQ catalog.
The first training run can build this embedding/SID cache itself, so the
data-only audit is sufficient before launching an experiment.

## 3. Train

```bash
bash experiments/amazon23_domains/run_train.sh music DIFF_GRM 0
bash experiments/amazon23_domains/run_train.sh music AR_GRM 0
```

`music` can be replaced by `video`, `science`, or `office`. `RUN_ID` overrides the
default run name. To train on an alternate text view:

```bash
ITEM_TEXTS_FILE=item_texts_rich_meta_reviews_top5.csv \
METADATA_CACHE_TAG=rich-reviews-top5 \
RUN_ID=music23_rich_reviews_diff_v1 \
bash experiments/amazon23_domains/run_train.sh music DIFF_GRM 0
```

The metadata cache tag prevents embeddings and semantic IDs from a different
text view from being reused accidentally.

## 4. Evaluate a checkpoint

```bash
bash experiments/amazon23_domains/run_inference.sh \
  music DIFF_GRM \
  saved/AmazonReviews2023CleanGR_music23_full_opq_cf_diff_guided_l20_v1/pytorch_model.bin \
  test 0
```

Inference disables the expensive decoder diagnostics by default. Set
`DIAGNOSTICS=1` to enable them. For an alternate text view, pass the same
`ITEM_TEXTS_FILE` and `METADATA_CACHE_TAG` environment variables used during
training.
