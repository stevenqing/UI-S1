# Amendment 004: Triple-Nested Checkpoint Selection

Date: 2026-08-11

Timing: frozen after the eligibility anchor passed, while implementing VUS-SR, before any VUS-SR model fit or result.

## Problem

Amendment 003 reserved one of the three inner-training folds for early stopping while also saying that the behavior policy was fitted on all three folds. That would let the checkpoint-validation fold influence its own fallback through CEV reliability/configuration, making early-stopping loss optimistic.

## Correction

For each outer fold `k` and OOF holdout fold `h`:

1. Let the three remaining outer-development folds be `T`.
2. Let checkpoint-validation fold `v` be the first cyclic fold after `h` that belongs to `T`.
3. Fit the VUS-SR model only on `F = T \ {v}` (two folds).
4. Fit one CEV behavior policy using only `F`. Apply that unchanged policy to model-training folds `F`, checkpoint-validation fold `v`, and OOF holdout fold `h`.
5. Fit structural reliability statistics only on `F`; use leave-one-row reliability for training rows and fixed `F` reliability for checkpoint/OOF rows.
6. Select the checkpoint epoch using only fold `v`. Emit OOF predictions only on fold `h` from the selected checkpoint.
7. Select S1/S2/S3 and safe thresholds from the four OOF holdout folds.
8. For final outer fit, train the selected configuration from initialization on all four outer-development folds for the median selected epoch from its four inner fits. Use the frozen outer-fold CEV policy and development-only reliability. Evaluate outer test exactly once.

The two-fold checkpoint-stage behavior policy is intentionally conservative. It prevents both the OOF holdout and checkpoint-validation fold from defining their own fallback. No VUS-SR grid, architecture, loss, gate, or outer-test rule changes.
