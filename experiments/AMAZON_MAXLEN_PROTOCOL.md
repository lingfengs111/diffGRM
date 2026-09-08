# Amazon max-history protocol

The default comparable protocol for every Amazon dataset in this repository is
`max_history_len: 20`. This matches Latte/PSID and the established Beauty14
setup. It applies to both DIFF_GRM and AR_GRM.

## Rules

1. New Amazon experiment configs and launchers default to 20 history items.
2. Keep `max_history_len` and the compatibility key `max_hist_len` equal.
3. A non-20 run must be intentional and use `MAX_HISTORY_LEN`. The standard
   launchers automatically include `_l<length>` in their run IDs; a custom
   `RUN_ID` should preserve that suffix.
4. The model length cannot exceed the cap stored in the prepared dataset's
   `stats.json`. Rebuild the splits first if a larger cap is required.
5. Existing L50 checkpoints/configs remain valid as legacy reproduction runs;
   do not compare them directly with Latte-L20 without labeling the difference.

The effective context is:

```text
min(prepared max_history_items, model max_history_len)
```

Amazon23 data prepared before this policy retains up to 50 items. It does not
need rebuilding for L20: the model takes the most recent 20. Future runs of the
CleanGR Amazon23 preparation script write a cap of 20. Amazon14 prepared data
already has a cap of 20.

## Standard launch

```bash
bash experiments/amazon23_domains/run_train.sh video DIFF_GRM 0
bash experiments/amazon14_domains/run_train.sh beauty DIFF_GRM 0
```

Both commands use L20. An explicit legacy/ablation run looks like:

```bash
MAX_HISTORY_LEN=50 \
  bash experiments/amazon23_domains/run_train.sh video DIFF_GRM 0
```

The Amazon dataset adapter logs a protocol warning for non-20 configurations
and rejects a requested model length larger than the prepared-data cap.
