# Allocation-Law Execution Spec

Date: 2026-08-01

Upstream: `runs/ccm-h2h/2026-07-31/` at commit `d9754844f7f563a4a30deeefbcb3a7137a059176`.

Status: preregistered before any Allocation-Law result.

## Claim

At fixed test-time forward budget, candidate-pool error correlation limits realizable scaling more strongly than the final similarity-based selection rule. Allocate forwards to lower failure correlation subject to a minimum source-quality gate.

## L1 budget curves

ScreenSpot-Pro, 1,581 rows. Budgets: 4, 8, 12, 16, 24 forwards.

- V-only: GTA1 full image plus official attention crops in rank order.
- Mixed: GTA1-7B, Qwen3-VL-8B-Instruct, and UI-TARS-7B-SFT under shared GTA1 region geometry. Per-budget source counts and ordering are frozen in `configs/l1_pools.yaml`.
- Aggregators: official MVP B3 and fold-local M1 CCM. Report pass@N.
- MDE: frozen H1 value 0.007043345177520599.

Availability: the H1 official generator returned 16-19 total unique candidates per row. V-only N=16 is available for all rows; N=24 is structurally unavailable and must be marked `TRUNCATED_UNAVAILABLE`, never padded or duplicated. Mixed N=24 is available by generating eight shared views for each of three models.

P-L1a and P-L1b are defined exactly in `configs/l1_predictions.yaml`. L-K1/L-K2 follow that file.

## L2 correlation-to-gain law

Construct fixed budget-12 pools from frozen full/view candidates according to `configs/l2_pools.yaml`. Failure kappa is computed only on outer development folds. Held-out outcomes are pass@12, B3, and M1.

Report Spearman correlation between development mean pairwise failure kappa and held-out scores across pools, plus 10,000 application-group bootstrap replicates. Prediction requires negative rho with absolute value greater than 0.7 for pass@12. B3/M1 correlations are secondary strengthening tests. This is a small-pool descriptive law and cannot support extrapolation beyond the frozen source set.

## L3 step-level extension

Mind2Web remains discovery-stage because no versioned corrected-label release exists. Fixed budget 12:

- V-only: TongUI-7B full plus 11 attention-proposed crops.
- Mixed: three result-blind deployable lineages, four views each.
- Lineage-only: 12 full-view models only if 12 eligible aligned candidates exist.

B3 migration changes only coordinates and inherits action/parameter from its selected source candidate. A5d-risk uses the frozen step-level implementation and nested grouped folds. Success requires Mixed > V-only and delta > 0.011558476230933909 (C1 Mind2Web MDE).

No L3 model or proposer choice may use L3 pooled outcomes.

## L4 proposer ablation

Equal 12-forward D2 pool:

- E1 shared GTA1 proposals (existing H3).
- E2 each model's native attention proposal pipeline.

No direction is preregistered. Before E2 pooled scoring, record each proposer's target-bbox-containing ratio and freeze layer/query-token settings. If a native proposer cannot be implemented under the released architecture without unsupported conversion, mark that model and E2 as unavailable rather than substituting a heuristic proposer.

## Kill conditions

- L-K1: P-L1a fails under both B3 and M1. Main budget-curve allocation claim dies; retain H3 as a narrow N=12 result.
- L-K2: P-L1b fails. Allocation and rule contributions must be stated jointly.
- L-K3: L2 pass@12 rho is nonnegative or abs(rho) < 0.7. Correlation law becomes parallel observations only.
- L-K4: L3 Mixed does not exceed V-only by Mind2Web MDE. Restrict allocation claim to pure grounding.

## Execution rules

- Fail closed on incomplete identities, unequal forward budgets, candidate duplication, fold leakage, source-quality violations, or target-field access in proposal/inference paths.
- Compact configs/code/result JSON/PDF/reports are tracked. Checkpoints, raw images, raw candidates, and shard logs are ignored.
- Existing H3 N=12 values are reused only when candidate set, order, folds, and aggregator contract are byte/semantically identical; otherwise candidates are reused and the point is reevaluated.
