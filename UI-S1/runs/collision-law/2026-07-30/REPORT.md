# Collision-Law Stage Report

Date: 2026-07-30

Status: W0 and W1 complete; independent E5 superseded by W2 with one compatible cell inherited; W2/W3/W4 pending.

## Preregistration record

The initial preregistration was pushed as commit `2aafdd2` before any Collision-Law result. Three result-free amendments were subsequently pushed before the corresponding final runs:

1. `c528dba`: separate Mind2Web GT-only analysis kernel from the GT-free inference kernel;
2. `420888f`: preserve released-parser out-of-domain points without clipping or parse repair;
3. `690e9d9`: correct A2 to discrete density mode and P1 to categorical same-error kappa after declaring the first uncommitted run invalid.

No invalid preliminary W1 JSON was committed.

## W0 data layer

W0 inherits the locked 102,054-row complementarity table and adds exactly:

- `view_id`, currently `full` for all inherited predictions;
- `pred_source`, currently `<model>__full`;
- `gt_element_area` for Mind2Web.

Both upstream analyzers were rerun and reproduced byte-identical summaries. The output contains 17 prediction sources. Four released Mind2Web identities have GT boxes outside the normalized viewport; their areas are preserved, flagged in the manifest, and never clipped.

## Kernel boundary

AndroidControl uses the exact metric-derived Gaussian coordinate kernel with `sigma=0.07`, plus the fixed uniform-cluster normalization `rho0=0.30712361963678886`.

Mind2Web parsers expose points but no predicted element boxes. Its exact point-in-GT-bbox evaluator can therefore be used only for post-hoc analysis and scoring. Inference PKA uses the fixed GT-free triangular point kernel on normalized-coordinate scale. This weakens the paper's exact evaluator-kernel claim on Mind2Web and is stated as a main-text limitation.

The operator gate has nine passing tests: identity at `K=1`, parse removal, parameter-free plurality, independent density-medoid agreement, continuous-mode behavior, GT-free API separation, out-of-domain preservation, and sequential type preservation.

## W1 main table

Primary scope is the preregistered double-sided deployability band. Values are held-out grouped-fold Step SR.

| Pool | A0 held-out best | A1 plurality + median | A2 plurality + density | A3 joint PKA | A4 continuous PKA | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| AndroidControl Low | 79.11% | 72.14% | 72.08% | 68.26% | 68.46% | 87.54% |
| AndroidControl High | 60.76% | 59.32% | 59.44% | 50.33% | 50.37% | 76.27% |
| Mind2Web visual | 51.78% | 57.12% | 58.22% | 58.41% | 32.69% | 79.86% |

### Operator deltas

| Pool | A2 - A1: density value | A3 - A2: joint value | A4 - A3: continuous value | A3 - A0 |
|---|---:|---:|---:|---:|
| AndroidControl Low | -0.07 pp | -3.82 pp | +0.20 pp | -10.85 pp |
| AndroidControl High | +0.12 pp | -9.11 pp | +0.04 pp | -10.43 pp |
| Mind2Web visual | +1.11 pp | +0.19 pp | -25.72 pp | +6.63 pp |

Density mode is useful on Mind2Web but not materially useful on AndroidControl. Joint product scoring adds only 0.19 pp over sequential density on Mind2Web and causes large AndroidControl regressions. Continuous mode is catastrophic on Mind2Web because density consensus does not imply target-element membership.

K3 is therefore triggered: joint PKA is not generally better than sequential density mode. PKA remains a unified perspective and a positive Mind2Web medoid result, not a generally superior operator.

K2 remains pending because the +6.63 pp Mind2Web A3 gain must be compared with W2's MDE.

## P1 strata

P1 uses categorical Cohen kappa over inherited exclusive error labels, conditioned on pairwise co-failure. Binary failure kappa is reserved for P3 lineage allocation.

| Stratum | Rows | Mean same-error kappa | A3 gain over held-out best |
|---|---:|---:|---:|
| Mind2Web CLICK | 1,774 | 0.211 | +8.23 pp |
| Mind2Web SELECT + TYPE | 306 | 0.166 | -2.61 pp |
| AndroidControl action-dominant hard core | 1,443 | 0.119 | 0.00 pp |
| AndroidControl grounding-dominant hard core | 930 | 0.050 | 0.00 pp |

The preregistered reverse-order prediction is not satisfied: Spearman correlation between collision and gain is `+0.316`, `p=0.684`. This is a valid negative result. The AndroidControl hard-core strata have zero A0 and A3 success by construction, limiting their usefulness for rank-correlation testing.

## Cross-lineage kappa

`w1_kappa.json` contains the preregistered binary failure kappa matrices with 1,000 matched-marginal permutations:

- Mind2Web: TongUI-7B, CogAgent-18B, UI-TARS-72B, and SeeClick;
- AndroidControl: all cross-family UI-AGILE, GUI-R1, and UI-R1-E pairs in Low and High.

These values are frozen inputs to P3 allocation; no W2 allocation result has been generated.

## E5 inheritance and W2 gate

E5 was paused at the user's request. The first new cell, AndroidControl `original_768`, has four complete 1,927-row shards and a complete merged 7,708-row prediction file. Its score process was interrupted, so scoring remains pending; no inference must be repeated.

Amendment 004 applies the superseding spec: independent E5 is canceled and its compatible cell is inherited as GUI-R1-7B / AndroidControl High / W2 `v4`. The remaining prompt-paraphrase E5 cells are not part of W2 and are canceled. W2's five views now provide the MDE, so W2 implementation and inference may proceed.

## Current scientific state

- P1: failed on the four preregistered strata.
- P2/K1: pending W2 views.
- P3: pending W2 model-view pool.
- K2: pending W2 noise floor.
- K3: triggered; operator demoted to unified perspective.
- AndroidControl aggregation: negative result across A1-A4.
- Mind2Web: discrete density and joint medoid aggregation are positive; continuous mode is strongly negative.