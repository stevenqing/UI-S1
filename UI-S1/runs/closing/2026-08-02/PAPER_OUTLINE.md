# Paper Outline: Fixed-View Lineage Allocation for GUI Grounding

Status: frozen writing skeleton; F3-triggered K8 X2 replacement remains pending and is confined to Section 7.

## 1. Problem and claim

GUI grounding test-time scaling has largely diversified candidates within one model through zoom, stochastic sampling, or learned selection. Our measurements identify lineage allocation as an orthogonal control variable.

Opening evidence:

- Same-model view failure kappa: 0.895.
- Cross-lineage failure kappa: 0.398.
- Same-family cross-scale kappa: 0.618.
- Fixed-view V-only budget slope is negative while Mixed is positive with non-overlapping 99% bootstrap CIs.

Primary claim wording:

> Under a fixed test-time forward budget, allocating candidates across model lineages can improve GUI grounding even when every added lineage is individually weaker than the original model.

Scope after F2: `fixed-view allocation axis`, not all single-model diversity. GTA1 sampling GUI-RC has a slightly negative point slope but its 99% CI crosses zero; sampling-plus-view B3 is significantly positive.

## 2. Mechanism

Two independent measurements establish the mechanism.

1. Failure correlation: candidate families differ mainly in correlated failure, not only marginal quality.
2. Proposal degradation: GTA1 proposal full-bbox containment declines from 99.94% at rank 0 to 61.04% at rank 11.

Interpretation: adding more fixed views from one lineage increasingly repeats correlated errors and lower-quality proposal ranks. Cross-lineage allocation trades some marginal candidate quality for lower joint failure correlation.

Do not claim the L2 correlation coefficient as a law: held-out fold-pool rho was -0.326, below the preregistered absolute 0.7 threshold.

## 3. Main result: weaker models make the pool stronger

Lead with the local N12 controlled comparison.

- GTA1-only M1: 60.40%.
- Qwen3-only M1: 56.80%, 3.61 points below GTA1.
- UI-TARS-only M1: 52.44%, 7.97 points below GTA1.
- Mixed M1: 63.82%.
- Mixed minus GTA1: +3.42 points, 99% CI [+1.41, +5.67], one-sided p=1e-4, 4.85 times MDE.
- Mixed minus Qwen3: +7.02 points, 99% CI [+4.65, +9.51].
- Mixed minus UI-TARS: +11.39 points, 99% CI [+8.30, +14.43].

Core sentence:

> The gain cannot be explained by adding a hidden stronger model: both added lineages underperform GTA1 in isolation, yet their candidates improve the combined pool.

The composition table tempers model-count claims: GTA1+Qwen3 M1 is 63.88%, statistically close to the three-lineage 63.82% point. Composition and correlation matter more than count.

## 4. Budget-sign reversal and sampling boundary

Fixed views:

- V-only M1 slope per forward: -0.002467, 99% CI [-0.004908, -0.000124].
- Mixed M1 slope: +0.003052, 99% CI [+0.001082, +0.005053].

Sampling F2:

- S-only GUI-RC N4/N8/N12/N16: 49.53/49.46/49.27/49.21%.
- Point slope: -0.000285, 99% CI [-0.000789, +0.000203]; sign not established.
- S-only B3 CI also crosses zero.
- Sampling-plus-view B3 slope is positive with 99% CI [+0.000719, +0.002477].

Conclusion: title and abstract must say `fixed-view lineage allocation`, not `single-model diversity axis`. Sampling and zoom/fixed-view mechanisms are empirically different.

## 5. Drop-in rules and confidence

Portable formulation:

> We do not modify the downstream rule; we replace only the candidate source.

Case 1, B3:

- GTA1-only: 60.09%.
- Mixed: 63.69%.
- Delta +3.61 points, 99% CI [+1.31, +6.22], p=1e-4.

Case 2, SafeGround:

- Stochastic GTA1 N4 AUROC: 0.628.
- V-only N12 AUROC: 0.744.
- Mixed N12 AUROC: 0.830.
- Frozen weights and implementation are unchanged. Deterministic N12 transfer does not inherit SafeGround's FDR guarantee.

Case 3, H1 graph centroid:

- V-only N12: 60.85%.
- Mixed N12: 63.50%.
- Delta +2.66 points, 99% CI [+0.48, +5.16], p=7.0e-4.

The unchanged pass@12 admission additionally rises 72.80% to 79.19%; this is an oracle/headroom diagnostic, not a deployable rule.

Attribution defense:

- Candidate-source replacement under B3 contributes +3.54 to +3.61 points depending on the frozen H3/L1 row serialization.
- Fold-local M1 adds only about +0.13 to +0.19 points over B3 inside the mixed pool.
- Same-candidate H1 M1 over B3 was +0.44 points, below MDE and not significant (p=0.148).

Thus the main gain is allocation, not reranker tuning.

Confidence mechanism closure:

- High-collision S_gap correctness AUROC is 0.393 on AndroidControl Low.
- Failed high-gap examples are enriched in exact hard cores by 1.97x, 2.70x, and 2.30x across the three C2 pools.
- Lower-correlation candidate pools make spatial dispersion useful again.

## 6. Deployment tool

A label-free pool feature, development mean pairwise normalized candidate distance, is fitted on 40 L2 fold-pool observations and validated on 10 new X2 fold-pool observations.

- Held-out Spearman: 0.903.
- p=3.44e-4.
- Training R-squared: 0.0145.

Interpretation: the result is useful for monotonic pool ranking but does not support a calibrated linear performance model. The validation is low-power because it contains only two new pools over five folds.

## 7. Negative results and boundaries

Selector failures:

- Same-candidate M1 does not significantly beat B3 and does not exceed the paper-only 62.8 selector reference.
- Collision likelihood-ratio selection is mixed against held-out A0: +0.56 points, p=0.071 on AndroidControl Low; -1.23 points, p=1.24e-4 on High.
- Naive stochastic sampling does not establish a budget slope.

Area mechanism:

- Smallest area quintile has Mixed pass@12 +6.62 points but M1 -2.52 points versus V-only.
- Largest quintile has M1 +8.23 points.
- Reject the hypothesis that small targets lack a correct mixed candidate. The bottleneck is headroom realization/selection on small targets in this setup.

F3/X2:

- Official Qwen2.5-VL-7B K8 UI-Zoomer reaches 40.04%, within 0.96 points of the reported 41.0 anchor.
- K3 reaches 36.69%, 3.35 points below K8; the original K3 X2 is length-sensitive and cannot evaluate official UI-Zoomer.
- The preregistered fixed-27 K8 paired rerun replaces the K3 X2 conclusion when complete. X2 never enters R1-R4.

Unavailable items, each one sentence:

- Scanner+Locator: no official implementation or auditable fixed-budget trace.
- Topology triangle: pure serial and full hybrid corners are absent.
- Mind2Web N6 same-lineage: family counts 1/3/2 cannot form N6.
- Original L3: required attention-proposal crops are unavailable; keep blocked.
- L4 E2: released proposer extraction is architecture-bound and unavailable for Qwen3/UI-TARS.

AndroidControl boundary:

- Aggregation ranking is metric-radius sensitive, especially at strict Low thresholds.
- High and Low results differ materially; no ScreenSpot allocation claim is transferred to AndroidControl.

## Abstract-ready contribution summary

1. We identify fixed-view lineage allocation as a missing test-time scaling axis using failure-correlation measurements.
2. At equal N12 compute, adding two individually weaker lineages improves M1 by 3.42 points with a positive 99% paired-bootstrap interval.
3. The gain transfers without rule changes to B3 and SafeGround-style confidence diagnostics.
4. Fixed-view budget slopes reverse sign under mixed allocation, while stochastic sampling does not show the same negative-slope law.
5. A label-free pool-distance signal predicts held-out pool ordering, with explicit low-power and calibration caveats.
