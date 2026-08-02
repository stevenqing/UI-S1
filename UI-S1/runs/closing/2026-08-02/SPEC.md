# Closing: Positive-Result Consolidation

Date: 2026-08-02

Upstream:

- `runs/collision-law/2026-07-30/`
- `runs/ccm-h2h/2026-07-31/`
- `runs/allocation-law/2026-08-01/`
- `runs/diversity-axis/2026-08-02/` at `cb872aa04ce542936b22d54a6506954713a7f824`

Status: result-free closing preregistration. This work introduces no new research direction.

## Positive claims

- R1: mixed lineage N12 exceeds every single-lineage N12 pool even though Qwen3 and UI-TARS are individually weaker than GTA1.
- R2: unchanged third-party rules improve when only the candidate source changes: B3 accuracy and SafeGround uncertainty AUROC.
- R3: under the same local backbones and test-time forward budget, mixed N12 exceeds all internally evaluated single-lineage configurations. Published 62.8 and 73.1 values are paper-only, non-comparable references and never enter a difference or significance test.
- R4: allocation changes the sign of the fixed-view budget slope. Sampling-family scope remains conditional on F2.

## F1 paired bootstrap

Reconstruct frozen held-out row outputs for mixed/v-only N12 and N16 and all three single-lineage N12 pools. Use 10,000 application-group bootstrap replicates stratified by the frozen outer fold, seed 20260802. Report paired delta, 99% percentile CI, and one-sided p-value `P(delta <= 0)`. MDE is 0.007043345177520599.

## F2 sampling coverage

Generate exactly 16 stochastic GTA1 full-image samples per ScreenSpot-Pro identity with temperature 0.5, top-p 0.95, and deterministic seeds. Evaluate N=4/8/12/16 prefixes with unchanged GUI-RC and B3. Mixed-sampling is unavailable because Qwen3/UI-TARS have no pre-existing matched random traces and this closing spec requests only GTA1 generation. The same-model sampling-plus-view cross-control alternates GTA1 sampling and frozen GTA1 official views, starting with sampling, at every prefix.

Fit N-to-accuracy slopes and 10,000 fold-stratified application bootstrap CIs as in X3. S-only GUI-RC is the primary sampling-family test; B3 and the sampling-plus-view control are secondary. If the primary slope CI is strictly negative, R4 may use "single-model diversity axis"; otherwise the title narrows to "fixed-view allocation axis". No padding or reuse.

## F3 UI-Zoomer anchor

Run official UI-Zoomer commit `2c1125067958df2468663004b2b4b7c50557da25` with local Qwen2.5-VL-7B-Instruct on ScreenSpot-Pro: K=8, temperature 0.9, gate 1.5, sigma 2.5, minimum crop 512. The reported method anchor is 0.410 and tolerance is absolute 0.01. Also run the official deterministic baseline (reported 0.276) and a K=3 microchain ablation on the same checkpoint.

Outcome handling is frozen in `configs/f3_outcomes.yaml`. F3 does not enter R1-R4.

## F4 area mechanism

Use the exact five area quintiles frozen by X3. Reconstruct N12 V-only and Mixed row-level pass@12. Report each pool's pass@12 and delta by quintile. The coverage-limited hypothesis is supported only when the smallest quintile's Mixed-minus-V-only pass@12 is nonpositive. No alternate binning is tested.

## Paper boundaries

- Do not claim absolute ScreenSpot-Pro SOTA.
- Paper-only values are visibly marked and excluded from calculations.
- "Drop-in" means rule and implementation unchanged, with only candidate source replaced.
- X4, X5, X8, original L3, and L4 E2 retain their upstream unavailable/blocked statuses.
- X2 treatment follows F3 only and never enters R1-R4.
- Checkpoints, raw traces, images, and logs remain untracked.
