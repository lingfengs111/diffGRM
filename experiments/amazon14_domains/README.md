# Amazon14 Beauty, CDs, Sports, and Toys

These configs use the canonical Beauty14 model and data protocol for all four
Amazon Reviews 2014 5-core domains. The new CDs, Sports, and Toys data has
already been prepared and audited through `AmazonReviews2014CleanGR`.
All runs default to `max_history_len=20`; see
[`../AMAZON_MAXLEN_PROTOCOL.md`](../AMAZON_MAXLEN_PROTOCOL.md).

| domain | interactions | users | items | train | validation | test |
|---|---:|---:|---:|---:|---:|---:|
| beauty | 198,502 | 22,363 | 12,101 | 131,413 | 22,363 | 22,363 |
| cds | 1,097,592 | 75,258 | 64,443 | 871,818 | 75,258 | 75,258 |
| sports | 296,337 | 35,598 | 18,357 | 189,543 | 35,598 | 35,598 |
| toys | 167,597 | 19,412 | 11,924 | 109,361 | 19,412 | 19,412 |

## Rebuild the new prepared datasets

From `/home/lingfengs111/codes/GR/CleanGR`:

```bash
bash scripts/prepare_amazon14_diffgrm.sh all prepare
```

Here `all` means CDs, Sports, and Toys; Beauty is only rebuilt when explicitly
requested. Alternate item-text views preserve the catalog and splits:

```bash
bash scripts/prepare_amazon14_diffgrm.sh sports rich_meta
bash scripts/prepare_amazon14_diffgrm.sh sports rich_reviews
```

The review view only uses deduplicated user-item reviews visible in the
training split, excluding validation/test reviews.

## Audit

```bash
python scripts/validate_prepared_protocol.py \
  --dataset AmazonReviews2014CleanGR \
  --config experiments/amazon14_domains/common.yaml \
  --config experiments/amazon14_domains/cds14.yaml \
  --data-only
```

Replace `cds14.yaml` with `sports14.yaml`, `toys14.yaml`, or `beauty14.yaml`.
Omit `--data-only` to also build and collision-audit the embedding/OPQ catalog.

## Train

```bash
bash experiments/amazon14_domains/run_train.sh cds DIFF_GRM 0
bash experiments/amazon14_domains/run_train.sh cds AR_GRM 0
```

The domain can be `beauty`, `cds`, `sports`, or `toys`. `RUN_ID` overrides the
default name. To use an alternate text view, set the same variables for both
training and inference:

```bash
ITEM_TEXTS_FILE=item_texts_rich_meta_reviews_top5.csv \
METADATA_CACHE_TAG=rich-reviews-top5 \
RUN_ID=sports14_rich_reviews_diff_v1 \
bash experiments/amazon14_domains/run_train.sh sports DIFF_GRM 0
```

## Evaluate a checkpoint

```bash
bash experiments/amazon14_domains/run_inference.sh \
  cds DIFF_GRM \
  saved/AmazonReviews2014CleanGR_cds14_full_opq_cf_diff_l20_v1/pytorch_model.bin \
  test 0
```

Inference disables decoder diagnostics by default. Set `DIAGNOSTICS=1` to
enable them.
