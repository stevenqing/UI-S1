# ICC: EVID premise audit

Round: `icc`

Date: 2026-08-15

Status: `PREREGISTERED_BEFORE_ANY_ICC_RESULT`

GPU: zero. All outputs are deterministic recomputations over frozen artifacts.

## Scope and correction

ICC is diagnostic only. It evaluates no method, creates no method claim, and changes none of F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, EVID, or XSOFT.

EVID's fixed constants came from AndroidControl failure kappas, not ScreenSpot-Pro error correlations. EVID therefore rejected its fixed transferred-constant variant, not the entire score family under correctly estimated ScreenSpot-Pro correlations. EVID is not modified or rescued inside ICC. This error is recorded in `docs/research_disclosures.md`.

All label-dependent outputs are evaluation-side. ScreenSpot-Pro remains post-selection.

## Inputs

Use the frozen 1,581-row, 12-candidate ScreenSpot-Pro C-uni bank in canonical view-major order. Source lineages are GTA1, Qwen3, and UI-TARS; views are 0-3. Bind EVID `STAGE0.json`, `STAGE1.json`, `SELECTED_PARAMETERS.json`, DECOMP `ARM1.json`, MASK `STAGE1.json`, and the complete views 0-5 bank required for the same-budget audit by SHA-256 before computation.

The five leaked aggregate ScreenSpot cells are prohibited as targets, thresholds, or selection criteria.

## Arm A: fitted-rho endpoint audit

Arm A is pure table extraction from EVID's committed `SELECTED_PARAMETERS.json`. For each outer fold report:

- selected $(\rho_v,\rho_\ell)$;
- whether either coordinate is at 0 or 1;
- selected inner-validation accuracy;
- the complete 6x6 inner-validation surface in the frozen ascending grid order;
- row/column maxima and finite differences from the selected cell toward lower and higher rho when neighbors exist.

Classify each selected fold as `low_endpoint`, `high_endpoint`, `mixed_endpoint`, or `interior`. No refit or new accuracy is computed.

## Arm B: destination of the 111 changed rows

Use EVID Stage-0 raw block identities and Stage-1 raw fixed/B3 correctness. A row is changed only when the fixed block tuple differs from the B3 block tuple. The six exhaustive outcome classes are:

1. unchanged and correct;
2. unchanged and wrong;
3. changed wrong-to-correct;
4. changed correct-to-wrong;
5. changed correct-to-correct;
6. changed wrong-to-wrong.

For changed rows, classify block-composition direction using candidate lineage counts:

1. `diversity_increase`: new represented-lineage count is larger;
2. `diversity_decrease`: new represented-lineage count is smaller;
3. `same_L_concentration_increase`: represented-lineage count is equal and the largest within-lineage count increases;
4. `same_L_concentration_decrease`: represented-lineage count is equal and the largest within-lineage count decreases;
5. `lineage_substitution`: represented-lineage count and maximum count are equal but the three-lineage count vector differs;
6. `composition_same`: lineage count vector is identical although block membership changed.

These classes are mutually exclusive and exhaustive. Report changed count, beneficial, harmful, correct-to-correct, wrong-to-wrong, net change, and each direction's count and wrong-to-correct rate. `diversity_increase` is the operational proxy for stronger cross-lineage consensus; concentration increase is the proxy for more same-lineage votes. Do not infer causal attribution beyond these composition changes.

Write one fsynced JSONL row per changed sample with old/new block, old/new representative, before/after correctness, and direction class.

## Arm C: direct ScreenSpot-Pro error dependence

Code candidate failure as 1 and success as 0. For each outer fold, estimate on the four outer-development folds only.

### Pairwise estimators

Primary dependence is Pearson phi, the ordinary Pearson correlation between two binary failure vectors. Cohen kappa is a secondary MASK-comparability statistic.

Within-lineage stratum: all 18 pairs formed by three lineages times the six view pairs within four views.

Cross-lineage stratum: all 48 pairs formed by three lineage pairs times four-by-four views.

For each fold and stratum:

- compute every prespecified pair;
- exclude pairs with zero variance from the primary phi mean and report their count;
- average valid pair values equally;
- report an undefined-as-zero sensitivity mean;
- compute kappa with MASK's definition and undefined-as-zero policy.

Report the five fold values, unweighted fold mean, and full fold range. Also report pair-level values in raw JSONL. Application-group bootstrap uses 10,000 replicates and 99% percentile intervals for the two phi means; resample applications within each outer-development fold, recompute all pair statistics, and preserve equal pair weighting.

### Effective-count diagnostics

For each fold form the complete empirical 12x12 phi matrix with diagonal one and undefined off-diagonal pairs set to zero. Compute

$$
N_{\mathrm{eff},\phi}=\frac{12^2}{\mathbf1^TR_\phi\mathbf1}.
$$

Recompute the analogous kappa-matrix $N_{\mathrm{eff},\kappa}$ exactly as MASK. Aggregate each across folds weighted by the corresponding held-out fold row counts; the kappa result must reproduce 1.5936767669403409.

Using each fold's mean valid phi values, compute the exchangeable two-level sensitivity model

$$
N_{\mathrm{eff},2L}=\frac{144}{12+36\hat\rho_v+96\hat\rho_\ell}.
$$

This is a separate structured approximation, not a replacement for MASK's empirical matrix.

Retrospective A2 support requires both:

$$
\frac{|N_{\mathrm{eff},\kappa}-N_{\mathrm{eff},\phi}|}{N_{\mathrm{eff},\phi}}\le0.10
$$

and

$$
\frac{|N_{\mathrm{eff},2L}-N_{\mathrm{eff},\phi}|}{N_{\mathrm{eff},\phi}}\le0.10.
$$

Otherwise A2 is retrospectively not supported. This does not change GRAN G-P8 from its historical `NOT_ADJUDICABLE_PREREG_UNDERDEFINED` status; ICC reports a new retrospective diagnostic only.

Always display AndroidControl reference values 0.895 and 0.398 and their signed differences from ScreenSpot-Pro phi estimates.

## Same-budget lineage audit

The DECOMP $2\to3$ contrast compares cell means at fixed budget and shared view count. The historical leave-UI-TARS comparison instead used GTA1+Qwen3 with six views each at N12 versus full three-lineage x four-view N12. It is not an 8-versus-12 comparison; it is equal nominal budget with different allocation geometry. Earlier prose that reads it as 8-versus-12 is corrected here.

Recompute four N12 pools on identical rows and folds:

- full 3x4, views 0-3;
- omit GTA1: Qwen3+UI-TARS, views 0-5 each;
- omit Qwen3: GTA1+UI-TARS, views 0-5 each;
- omit UI-TARS: GTA1+Qwen3, views 0-5 each.

Report canonical B3 and fold-local source-priority F1 majority. For each omitted lineage report `full 3x4 minus 2x6` with 10,000 application-group paired bootstrap replicates and 99% CI. This audit explains composition-specific saturation and does not replace DECOMP's equal-cell $2\to3$ estimand.

## Interpretation

The three diagnostics are independent. A human final interpretation must cite all three.

- Lower ScreenSpot-Pro phi than transferred constants means EVID used overly strong discounting and did not test the score family at a direct ScreenSpot-Pro parameterization. Any follow-up must be separately named, post-selection, and explicitly cite EVID's failure.
- Similar phi and transferred constants means EVID failed near the relevant dependence scale.
- If Arm A selects low rho and Arm B's diversity-increase correction rate does not exceed concentration-increase correction rate, the direction closes regardless of Arm C.
- Any two diagnostics favoring closure are sufficient to recommend closure, but this is a recorded human interpretation rather than an automatic gate.

## Execution and retention

Commit this spec and `configs/icc_prereg.yaml` before implementation or result. Then run Arm A, Arm C, Arm B, and the same-budget audit in that order. Each implementation is committed before its result.

Raw JSONL uses write/flush/fsync. Inputs, outputs, disclosure, failed attempts, and dataset snapshots receive SHA-256 manifests and independent retention under `/scratch/workspaceblobstore/icc/2026-08-15`.