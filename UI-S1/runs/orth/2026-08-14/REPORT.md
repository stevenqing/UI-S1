# ORTH Orthogonal-Evidence Scoping Report

Date: 2026-08-14

Status: `ORTH_COMPLETE_PREREGISTER_OCR_CONFIRMATORY`

## Scope

ORTH is exploratory scoping only. No result is eligible for a paper claim, method claim, runtime rule, or change to any existing project status. A later confirmatory round must regenerate any claim-eligible evidence.

## Arm 0

CEIL's recoverable counts are unique arm-expanded sample keys, not candidate counts. Mind2Web has 2,021 recoverable sample keys but 891 unique base rows; ScreenSpot-Pro has 968 sample keys and 430 base rows. CEIL's primary interval was already group-clustered. The base-row-clustered sensitivity leaves the Mind2Web lower bound above 0.65, while the IID candidate-pair interval is much narrower and anti-conservative.

## Arm 1

Two independently implemented CPU OCR engines were run over all 1,581 ScreenSpot-Pro screenshots. The report uses matcher-family ranges over the complete frozen grid rather than selecting a best engine or threshold. Full per-setting results are in `ARM1.json`; range tables are in `MAIN_TABLES.md`.

OCR is evaluated separately on 977 text targets and 604 icon targets and projected onto selected-correct, recoverable, and zero-coverage row classes. All accuracy, overlap, and kappa values are evaluation-side.

## Arm 2

The official 2,094-action Mind2Web HTML lane was historically downloaded and completely audited, but its local dataset and candidate-score files are now absent. The current 2,080-row XFER lane has no full DOM/AX tree and retains only GT-selected positive snippets for 1,975 rows. No DOM predictor metric is computed; restoring and hashing the official data is a prerequisite.

## Arm 3

Marginal channel accuracy and error kappa identify a joint 2-by-2 error table, disagreement mass, and oracle selector headroom, but not Bayes-fused grounding accuracy. Visual weights 12 and 1.5937 both retain the visual channel on every disagreement when no row-level confidence exists. A confirmatory fusion study must define a common candidate space and calibrated per-candidate likelihoods.

## Scoping decision

Direction: `PREREGISTER_OCR_CONFIRMATORY`.

Across the full exploratory grid, both CPU OCR engines show substantial normalized/edit match coverage on recoverable and zero-coverage rows, and overall OCR-vs-pool error kappa remains roughly 0.10-0.20, below the 0.398 cross-family reference. The localization signal is concentrated on text targets (EasyOCR roughly 25-31% all-row accuracy; RapidOCR roughly 17-27%), while icon accuracy remains below 1.4%. This supports a narrowly scoped text-target confirmatory design, not a general OCR method. Full DOM/AX evaluation is deferred because the historically audited official dataset is currently missing locally, and Arm 3 shows that marginal accuracy/kappa alone do not identify a fusion policy without row-level calibrated evidence.

This direction is a design recommendation only. It does not authorize a paper result or modify CEIL/SPLIT/MASK/TRIVUS/VUS-SR.
