# LSA No-Action Cross-Arm Confirmation

Date: 2026-08-10

Status: `FROZEN_BEFORE_CONFIRMATION_RESULTS`

Discovery upstream: `runs/lsa/2026-08-10/`

## 1. Purpose

The preregistered no-action ablation was the strongest LSA variant on C-uni: Mind2Web improved over CEV-A by 1.73 pp with a positive 99% CI, while ScreenSpot-Pro was nearly unchanged. Because no-action was not the preregistered main model, this study does not re-test C-uni. It confirms or rejects transfer to all three untested learned-selector candidate pools: C-cond, C-rand, and C-self.

## 2. Frozen model

- Feature set: exact `no_action` indices from `lsa_variants.py`.
- Estimator: H3 in every outer fold.
- Outer-fold thresholds: 0.19777147655873645, 0.18570657406035798, 0.26659044214294886, 0.2094857223674213, 0.2762614019099612.
- Training candidates: C-uni outer-development rows only.
- Test candidates: each of C-cond, C-rand, C-self outer-test rows.
- No target-arm labels train the estimator or tune thresholds.
- Target candidate sources absent from C-uni use C-uni train-only lineage-average reliability. Shared source keys use exact C-uni train-only reliability.
- Baselines: frozen per-arm CEV-A and nested dev-selection outputs.

The model, features, thresholds, and fallback outputs cannot change after confirmation results.

## 3. Statistics

All six arm × benchmark cells are reported. Use 10,000 paired grouped bootstrap resamples and 99% percentile intervals. Also compute, within each benchmark, the average safe-minus-baseline effect across the three arms inside each bootstrap replicate.

## 4. Confirmation gates

### T1: cell safety

Every arm × benchmark cell must be non-inferior to CEV-A: CI upper bound nonnegative or absolute loss below benchmark MDE.

### T2: Mind2Web transfer

The equal-arm mean `LSA-no-action-safe − CEV-A` on Mind2Web must have 99% CI lower bound above zero.

### T3: ScreenSpot neutrality

The equal-arm mean on ScreenSpot-Pro must be non-inferior and its point loss, if any, must be below MDE 0.70 pp.

### T4: dev-selection comparison

The equal-benchmark/equal-arm standardized mean versus nested dev-selection must have 99% CI lower bound above zero, with no individual cell losing more than MDE.

## 5. Outcomes

- `CONFIRMED_SAFE_LEARNED_AGGREGATOR`: T1–T4 pass.
- `CONFIRMED_VS_CEV_ONLY`: T1–T3 pass but T4 fails.
- `PARTIAL_TRANSFER`: T1 passes but T2 or T3 fails.
- `FAILED_CROSS_ARM_CONFIRMATION`: T1 fails.

No arm may be omitted. C-cond is not privileged over the two mandatory controls.

## 6. Kill conditions

| ID | Trigger | Consequence |
| --- | --- | --- |
| LT-K1 | Candidate identity/count mismatch | Stop and debug |
| LT-K2 | Any arm × benchmark fails T1 | No learned aggregator claim |
| LT-K3 | Mind2Web equal-arm CI crosses zero | Discovery does not transfer robustly |
| LT-K4 | ScreenSpot equal-arm loss exceeds MDE | Universal safety fails |

Only LT-K1 permits debugging.
