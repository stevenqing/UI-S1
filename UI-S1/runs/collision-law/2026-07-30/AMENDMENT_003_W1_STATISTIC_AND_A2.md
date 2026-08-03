# Preregistration Amendment 003: W1 Collision Statistic and A2

Date: 2026-07-30

Status: applied after an uncommitted full-run implementation audit and before any W1 result is committed or reported as final.

## Invalid preliminary run

The first uncommitted W1 run is declared `INVALID_IMPLEMENTATION_NOT_REPORTED` for two reasons:

1. A2 called the A4 continuous mean-shift implementation instead of a discrete density mode, so A2 did not isolate density mode from joint product aggregation.
2. P1 used binary failure kappa inside AndroidControl strata defined by `all_models_fail=true`. Both model failure vectors were constant one, forcing kappa to one and making the statistic non-identifying.

The preliminary JSON files are overwritten after correction and are never committed.

## Corrected A2

A2 keeps plurality action selection, then selects the coordinate candidate with maximum coordinate-kernel density among predictions with the winning type. It is a discrete KDE medoid and uses no continuous mean-shift. A4 alone uses continuous mode.

Thus:

- A2 minus A1 isolates discrete density mode versus geometric median.
- A3 minus A2 isolates joint product scoring versus sequential type-first scoring.
- A4 minus A3 isolates continuous coordinate mode versus candidate medoid.

## Corrected P1 collision statistic

P1 measures **same-error collision**, not binary co-failure incidence:

1. For each model pair and stratum, retain rows on which both models fail.
2. Compare their inherited mutually exclusive `err_label` values with categorical Cohen kappa.
3. Generate a 1,000-permutation matched-marginal null by permuting one model's labels within the co-failure rows.
4. Average observed pairwise categorical kappa within the stratum.

Pairs with fewer than 30 co-failure rows are reported but excluded from the stratum mean. A degenerate pair with one identical category in both vectors is assigned kappa one; a degenerate pair without observed agreement is assigned zero.

Cross-lineage P3 allocation continues to use D3's binary failure kappa on the full clean pool. The two statistics answer different questions and are stored separately.

## Fixed P3 five-model selection

The double-sided band admits six Mind2Web models. The fixed five-forward C2 budget selects the five eligible models with highest full-clean-pool Step SR, breaking ties lexicographically. Under the locked upstream traces these are TongUI-7B, TongUI-32B, CogAgent-18B, TongUI-3B, and UI-TARS-72B. UI-TARS-7B remains eligible for W1 deployable-scope reporting but is outside the five-forward P3 corner.