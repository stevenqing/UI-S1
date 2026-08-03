# Amendment 002: Evaluation Operations

Date: 2026-08-01

Status: frozen after production view generation started and before any production shard was loaded, scored, or aggregated.

This amendment resolves implementation details left implicit in the result-free L1/L2 configs. It does not change candidate allocation, budgets, models, folds, metrics, MDE, or predictions.

## Candidate merge

Ground truth is loaded only from the GTA1 source at evaluation time and is not present in new model traces. All three model families are joined by the 1,581 frozen identities. Every `(model, view_index)` is unique, and every region must equal the corresponding region in the frozen N12 manifest. Existing model views 0-3 retain their N4 prefix hash; extended views 4-11 retain the N12 prefix hash. Hashes are checked against their own source contracts and regions are checked directly against N12.

GTA1 candidates retain official proposal coverage for B3. Qwen3 and UI-TARS candidates use coverage zero, matching the completed H3 evaluation contract. B3 and M1 are imported from the completed H3 implementation; no selector is reimplemented.

## L1 decisions

The common adjacent budget intervals are `4->8`, `8->12`, and `12->16`. P-L1a is evaluated separately for B3 and M1. A rule passes when at least one common interval has V-only increment strictly below MDE and Mixed increment strictly above MDE on that same interval. Full P-L1a requires both rules; one rule is a partial pass. The unavailable V-only N24 point is excluded rather than truncated or imputed.

For P-L1b at each jointly available budget, define `g_B3 = Mixed_B3 - V_B3` and `g_M1 = Mixed_M1 - V_M1`. The sign condition requires both gaps to be strictly positive or both strictly negative. The magnitude condition is `abs(g_B3 - g_M1) < min(abs(g_B3), abs(g_M1))`. P-L1b passes only if both conditions hold at every jointly available budget.

## L2 operations

There are exactly eight frozen pools in `configs/l2_pools.yaml`, not nine. Each has 12 unique `(model, view_index)` units. For each outer fold and pool, binary candidate failure vectors are computed on development rows only. Cohen kappa is computed for every candidate pair and averaged without test-row access. Constant-vector pair kappa is reported as null and excluded from the mean, with counts reported.

Each pool contributes five observations: one per outer fold, pairing development mean pairwise kappa with that fold's held-out pass@12, B3, or M1 accuracy. The primary Spearman statistic therefore uses 40 fold-pool observations. The eight-pool aggregate means are also reported descriptively but are not substituted for the preregistered held-out observations.

The 10,000-replicate application-group bootstrap resamples application groups, rebuilds each fold's development kappa records and held-out binary outcome records from sampled groups, and recomputes the 40-observation Spearman statistic. Fold-local CCM selectors are fit once on the original outer development folds; their held-out binary outcomes are resampled, not refit inside each bootstrap replicate. Seed is `20260801`. Non-finite replicates are counted and excluded from interval quantiles; the analysis fails closed if fewer than 99% are finite.

The original application-to-outer-fold mapping remains fixed during bootstrap. A sampled application's multiplicity applies to its frozen fold; applications are not reassigned across folds inside a replicate.

The 1,000 matched-marginal permutations use fixed candidate failure marginals. Sampling the pairwise failure overlap from its exact hypergeometric distribution is treated as computationally equivalent to randomly permuting one binary failure vector. For each fold-pool, the permutation p-value compares observed mean pairwise kappa with the null mean across the same nonconstant pairs. This diagnostic does not alter the primary Spearman input.
