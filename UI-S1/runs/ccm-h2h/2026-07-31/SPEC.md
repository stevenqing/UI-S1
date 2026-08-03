# CCM Head-to-Head Execution Spec

Date: 2026-07-31

Upstream: `runs/collision-law/2026-07-30/`

Status: preregistered before H1/H3 result generation.

## Claims

- H1: on identical ScreenSpot-Pro candidate sets and equal forwards, coordinate-domain CCM outperforms official MVP densest-cluster selection.
- H2: same-model view perturbations have a higher error-collision floor than cross-family replacements.
- H3: only if H2 is positive, an equal-forward mixed model/view pool outperforms a pure-view pool.
- H4: zoom benefit is ordered by target-to-screen area ratio across benchmarks.

## Pre-result corrections

- C1 reports both the original five-view MDE and an exchangeable `full`/`v1` estimate. For two observations, `MDE_v1 = 2 * sample_sd(full,v1) = sqrt(2) * abs(v1-full)`. Views v2-v4 remain distribution-shift diagnostics and are excluded from the main noise floor.
- C2 tests whether failed high-`S_gap` overrides are enriched in the all-model hard core relative to the hard-core base rate, with an exact hypergeometric upper-tail test.
- C3 replaces categorical same-error kappa with the fold-local CCM error-conditional agreement mass on non-hard-core/disagreement rows.

## H1 protocol

Model: pinned GTA1-7B revision already used by W3.

Dataset: pinned ScreenSpot-Pro, 1,581 rows.

Candidate counts: `N in {2,4,10}` means one full-image prediction plus `N-1` official attention-guided subimage predictions. The official source receives `--max_inferences N-1`; all other generation settings remain fixed to the validated W3 launch: attention layer 20, target token comma, batch size 1, original processor and prompt.

The existing official W3 trace has exactly five candidates (`max_inferences=4`) and is an anchor only. It is not reused as N=4 or N=10.

Grouped evaluation uses five folds keyed by `application`; if any application is absent, fail closed. Candidate files store id, application, image size, target bbox, ordered predictions, coverage, region, stage, and generation order. Aggregators assert identical per-row candidate hashes.

Baselines:

- B0 full image.
- B1 seeded random candidate, seed 20260731.
- B2 arithmetic coordinate mean.
- B3 official MVP complete-link/highest-coverage candidate.
- B3-paper complete-link centroid.
- B3-graph graph-centroid ablation.
- B4 algorithm-level ReGUIDE two-stage KDE; no RL or view-consistency training claim.
- M1 coordinate-domain CCM.
- M2 CCM plus nested risk-controlled fallback.
- pass@N oracle.

Coordinate CCM has one candidate class and one pair type in H1. Source prior is constant and omitted. Similarity uses normalized Euclidean distance transformed to `u = exp(-d^2 / (2*h^2))`, with `h` fixed from the evaluator scale. Rank-quantile likelihood-ratio bins make the decision invariant to strictly monotone reparameterization. Eight equal-frequency bins and add-one smoothing are frozen. Candidate success is point-in-target-bbox and is used only on development folds.

MDE uses three deterministic candidate-generation perturbation seeds. If official attention proposals are deterministic, seeds perturb top-k selection only; no target information enters proposal generation. `MDE = 2 * sample_sd` of aggregate accuracy.

Predictions:

- P-H1a: at N=10, M1 > B3, delta > MDE, paired bootstrap one-sided p < 0.01.
- P-H1b: M1 is non-decreasing over N=2/4/10; M1 N10-N4 is not significantly negative; B3 reproduces an N4-to-N10 decline.
- P-H1c: M1 > 62.8% and delta over 62.8% > MDE. This is explicitly a paper-only comparison to the published GRPO selector.

Bootstrap: 10,000 application-group resamples, seed 20260731. The observed point estimate and one-sided probability of delta <= 0 are reported.

## H2 protocol

Primary same-model view pairs are only `(full,v1)`. Views v2-v4 are excluded as non-exchangeable distribution shifts.

- View axis: GUI-R1-7B and UI-AGILE-7B, Low and High, paired full/v1 failure vectors.
- Cross-family axis: W1 full-view cross-family model pairs within each pool.
- Same-family scale control: W1 same-family model pairs.

Statistic: binary failure Cohen kappa plus 1,000 matched-marginal permutations, seed 20260731. The primary directional test compares the mean of the four AndroidControl view-axis kappas with the mean AndroidControl cross-family kappa. Its permutation null independently permutes pair labels while preserving group sizes; exact enumeration is used if feasible, otherwise 100,000 permutations.

H2 is positive only if view-axis mean kappa > cross-family mean kappa and one-sided p < 0.01. H3 is blocked otherwise.

Mind2Web has only one model with views and therefore contributes descriptive same-model-view evidence, not a matched cross-family primary test.

## H3 gate and pool rule

H3 may run only when `h2_collision_floor.json` marks H2 positive.

Equal budget: 12 forwards per row.

- D1 GTA1-7B, 12 official views.
- D2 GTA1-7B, Qwen3VL-8B, and one frozen third model, four views each.
- D3 12 models, one full view each, only if 12 eligible checkpoints exist.

Eligibility is frozen before H3 inference: a model's independently measured ScreenSpot-Pro bare accuracy must be at least half the GTA1-7B bare accuracy. Among eligible available models, select by descending bare accuracy, then model id. No H3 pooled result may alter selection.

Every pool reports B3 and M1. Success requires D2 M1 > D1 M1 and the delta exceeds ScreenSpot-Pro MDE.

## H4 protocol and availability gate

For each row, `x = log(screen_area / target_bbox_area)`. Outcome is zoom-based method correctness minus bare correctness. Report equal-count x bins and row-level Spearman correlation, with application-group bootstrap confidence intervals.

Primary success requires rho > 0.7 and positive direction on all three independently auditable benchmarks.

Availability audit before result generation:

- ScreenSpot-Pro: target bbox and image size available for all 1,581 rows.
- Mind2Web: normalized target bbox area available for all 2,080 visual rows.
- AndroidControl W2: released rows contain point GT only; no target bbox/area. Exact SHA256 linking to pinned AndroidControl-Curated images matched 0/7,708 rows. W4 bboxes therefore cannot be transferred to W2.

Consequently the preregistered three-benchmark H4 primary claim is currently blocked and H-K4 triggers unless an auditable versioned AndroidControl bbox mapping is obtained before H4 result generation. ScreenSpot-Pro and Mind2Web two-benchmark analyses remain descriptive and cannot satisfy H4.

## Kill conditions

- H-K1: P-H1a and P-H1c both fail. Remove same-candidate-set general method claim.
- H-K2: P-H1b fails and B3 N4-to-N10 decline is not reproduced. Move direct collision-saturation mechanism to H2; if H2 also fails, move mechanism section to appendix.
- H-K3: H2 negative. Do not run H3.
- H-K4: H4 rho < 0.7 or any of the three benchmark area axes is unauditable. Downgrade resolution law to descriptive evidence.

## Execution constraints

- Fail closed on incomplete identities, candidate count, candidate hash mismatch, target-field access in generation, or fold leakage.
- GPUs 0-7 may be used only when no protected external process occupies them.
- PID 1814 is external and must never be killed, paused, or reused. H1/H3 wait while it occupies the GPUs.
- Compact code/config/result JSON and reports are tracked. Model assets, images, raw candidates, and raw traces remain ignored.
