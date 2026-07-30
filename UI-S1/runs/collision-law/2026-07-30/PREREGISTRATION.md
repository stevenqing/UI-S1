# Collision-Law Preregistration

Date: 2026-07-30

Base commit: `802940669dc80f3dee572cf39643e867844d9965`

This document and the files under `configs/` are committed before any Collision-Law W1 or W2 result is generated. Existing overlap/complementarity results are treated as upstream observations, not Collision-Law test results.

## Claims

- P1: stratum-level aggregation gain is reverse ordered by chance-corrected failure collision.
- P2: under view perturbation, grounding flips exceed action-type flips; under model replacement, both are high.
- P3: at a fixed five-forward budget, cross-lineage-kappa-guided mixed allocation exceeds both corner allocations.

## Kill conditions

- K1: if P2 fails, the mechanism claim is withdrawn; the paper becomes measurement plus law and PKA moves to the appendix.
- K2: if Mind2Web density-mode total gain is below the W2 MDE, the aggregate positive result is withdrawn; element-miss stratum results may remain.
- K3: if joint PKA does not exceed sequential density mode, the unification property remains but the new operator is demoted to a unified perspective.

## Fixed W1 contract

- Main PKA weights are uniform.
- Main coordinate kernel on AndroidControl is Gaussian with `sigma = 0.07`, derived from half the evaluator radius `0.14`.
- Main Mind2Web coordinate kernel is the evaluator indicator for point-in-bbox.
- String kernel is the lane scorer's token-set F1.
- Type kernel is exact equality.
- Candidate set is the parsed predictions themselves; parse failures are excluded.
- Accuracy weighting, smooth Mind2Web boundary kernels, continuous mode, and dev-fold temperature are ablations only.
- Test paths must assert that no tunable kernel parameter is supplied.

Amendment 001 supersedes the Mind2Web inference-kernel sentence above after trace-schema inspection proved that parsers expose points but no predicted bboxes. The GT indicator remains analysis-only; inference uses the fixed unit-square triangular kernel documented in `AMENDMENT_001_MIND2WEB_COORD_KERNEL.md`.

Degeneracy tests required before W1 execution:

1. `K=1` is identity.
2. Parameter-free actions reduce to plurality.
3. Same-type coordinate predictions agree with the independently implemented density-mode reference within the declared candidate/continuous-mode distinction.

## Fixed W2 contract

Models:

- Mind2Web: TongUI-7B.
- AndroidControl: GUI-R1-7B and UI-AGILE-7B. Both are fixed in the P2 model axis before inference.

Views:

- `full`: existing original prediction, never rerun for the main matrix.
- `v1`: 28-pixel black border on all four sides.
- `v2`: prediction-centered crop with width and height equal to 50% of the original image.
- `v3`: prediction-centered crop with width and height equal to 75% of the original image.
- `v4`: deployment resize profile.

The crop center is clamped only by image bounds; missing regions are filled with black, and the crop is resized back through the lane's original processor contract. Crop generation must fail closed if any GT field is read. GT coverage is measured only after view generation and cannot alter a crop.

Decoding remains each lane's released greedy contract and original prompt. W2 noise uses view variation only; this directional limitation must be reported.

P2 primary comparison:

- action flip: `pred_action(view) != pred_action(full)`;
- grounding flip: predicted type unchanged, grounding-success indicator changes;
- report paired rates with confidence intervals by benchmark, setting, and GT-area bin.

Mind2Web normalized GT-area bins are fixed as `tiny <= 0.001`, `small in (0.001, 0.005]`, and `regular > 0.005`. AndroidControl has point labels rather than element boxes and is not assigned an area bin.

K1 is first evaluated using `full` versus `v1`. Later views strengthen or qualify the plot but do not redefine K1.

## Fixed P3 budget

Budget is exactly five forwards per row.

- C1: one model, five views, density mode.
- C2: five models, one view, PKA.
- C3: greedy model-view allocation minimizing average preregistered cross-lineage kappa to the selected pool.
- C4: random mixed allocation using seed `20260730`.

Evaluation uses the existing grouped folds in `runs/complementarity/2026-07-30/folds.json`.

Amendment 003 corrects the fixed Mind2Web C2 set to the top five eligible models by full-clean Step SR: TongUI-7B, TongUI-32B, CogAgent-18B, TongUI-3B, and UI-TARS-72B. The fixed AndroidControl C2 set is the five unified lanes already present in the tidy table.

## Execution gate

No W1/W2 result script may run until:

1. E5 reaches `PASS` and its transition table is frozen.
2. `configs/strata.yaml` and `configs/bands.yaml` are committed and pushed.
3. W0 reproduces upstream summary hashes and validates extended-row identity coverage.

W4 Curated work may perform availability/preflight checks before E5, but may not emit `w4_curated.json` or `w4_threshold.json` before this preregistration commit.