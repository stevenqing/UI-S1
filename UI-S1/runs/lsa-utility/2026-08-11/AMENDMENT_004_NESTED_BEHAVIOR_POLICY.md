# Amendment 004 — Nested Behavior Policy for Utility Labels

Date: 2026-08-11

Status: `PRE_RESULT`

Using the globally stored upstream OOF CEV policy for every Utility-LSA training label creates second-level stacking leakage: Utility outer-test fold labels helped fit CEV policies for rows in the other folds. Although each CEV output is OOF with respect to its own row, it is not nested OOF with respect to the meta-learner's test fold.

Before any Utility-LSA result, the behavior policy is corrected:

## Inner OOF

For Utility outer fold $k$ and inner holdout fold $h$:

1. The CEV behavior policy uses only the remaining three folds $S$.
2. The first fold in cyclic order after $h$ that belongs to $S$ is its CEV configuration-validation fold.
3. Source reliability and Mind2Web coordinate scale are fitted on the other two folds.
4. Select one CEV-global granularity/threshold tuple using the validation fold and the frozen CEV grid/tie order.
5. Refit reliability/scale on all three folds in $S$.
6. Apply that fixed policy both to Utility model-training rows in $S$ and inner holdout rows $h$.

The behavior policy never uses the inner holdout or Utility outer-test fold to select configuration/reliability.

## Final outer evaluation

For outer test fold $k$, use the exact frozen upstream CEV-A global configuration for fold $k$, refit source reliability on the other four folds, and apply the same fixed policy to Utility training rows and outer-test rows. The reconstructed outer-test correctness must match frozen CEV outputs row by row.

All upstream CEV-A outer folds selected the global variant, so excluding action-conditional behavior from the nested reconstruction is implementation-equivalent at final evaluation and avoids adding an unregistered inner action-selection layer.

No reward, Utility-LSA feature, model, threshold, or gate changes.
