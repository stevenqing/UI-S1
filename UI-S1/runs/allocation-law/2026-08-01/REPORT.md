# Allocation-Law Results

Date: 2026-08-01

- Preregistration: `00aa688`
- Result-free view extension: `d93d800`, `7b018be`
- Result-free L1/L2 evaluator: `5010456`
- Trace integrity amendment: `4d089eb`
- Bootstrap amendment: `21809c0`
- L3/L4 availability protocol: `9fe0ff8`

## Executive result

The strongest supported result is a budget-dependent candidate-allocation effect on ScreenSpot-Pro. Mixed-lineage allocation materially outperforms GTA1-only views at N=8, N=12, and N=16 under both B3 and M1, reaching about +5.5 pp at N=16. P-L1a passes under both rules and L-K1 does not trigger.

The stronger claim that allocation dominates rule choice at every common budget fails because N=4 has opposite-signed B3 and M1 allocation gaps. L-K2 triggers. The preregistered correlation law also fails its effect-size threshold: failure kappa has the expected negative association with held-out pass@12, but observed rho is only -0.326 rather than below -0.7. L-K3 triggers, so kappa and performance are reported as parallel observations rather than a validated law.

L3 is not evaluated because none of its fixed-budget pools can be constructed under the preregistered semantics. L4 E2 is unavailable because Qwen3 and UI-TARS do not expose released native attention proposers compatible with the required extraction path. No heuristic proposer or candidate padding is substituted.

## Integrity gate

The production extension generated Qwen3 and UI-TARS views 4-11 for all 1,581 ScreenSpot-Pro identities. Each model has four shards with 396/395/395/395 rows. The frozen N12 manifest SHA-256 is `2a7233cf0fbab109ea481ba891d2d7e65a481c48cd73c623d7d76fc61c06ae17`.

Before scoring, the loader verified every prediction hash, model revision, stable index, shard assignment, view index, region, finite point, and identity. Extended traces contain no target bbox. GTA1 has 16-19 unique candidates per row; V-only N24 is therefore unavailable and is never padded.

## L1: budget curves

Accuracy is step/row micro accuracy. Values are percentages.

| Pool | N | B3 | M1 | pass@N |
|---|---:|---:|---:|---:|
| V-only | 4 | 61.23 | 61.42 | 68.88 |
| Mixed | 4 | 61.86 | 59.90 | 73.43 |
| V-only | 8 | 60.72 | 60.78 | 71.22 |
| Mixed | 8 | 61.99 | 63.19 | 77.36 |
| V-only | 12 | 60.09 | 60.40 | 72.80 |
| Mixed | 12 | 63.69 | 63.82 | 79.19 |
| V-only | 16 | 58.32 | 58.25 | 74.07 |
| Mixed | 16 | 63.76 | 63.76 | 80.20 |
| V-only | 24 | unavailable | unavailable | unavailable |
| Mixed | 24 | 62.56 | 64.07 | 81.72 |

P-L1a passes for both rules, although at different breakpoints:

- B3: V-only decreases by 0.63 pp while Mixed increases by 1.71 pp from N=8 to N=12.
- M1: V-only decreases by 0.63 pp while Mixed increases by 3.29 pp from N=4 to N=8.

P-L1b fails. At N=4, Mixed minus V-only is +0.63 pp for B3 but -1.52 pp for M1. At N=8, N=12, and N=16 both gaps are positive; at N=16 they are +5.44 pp and +5.50 pp respectively.

## L2: correlation-to-gain

Eight frozen N12 pools produce 40 outer-fold/pool observations. Kappa is computed only on outer development rows; B3, fold-local M1, and pass@12 are held out.

| Pool | Dev failure kappa | pass@12 | B3 | M1 |
|---|---:|---:|---:|---:|
| GTA1 12 views | 0.689 | 72.80 | 60.09 | 60.40 |
| Qwen3 12 views | 0.712 | 74.19 | 56.93 | 56.80 |
| UI-TARS 12 views | 0.676 | 70.02 | 51.87 | 52.44 |
| GTA1 + Qwen3, 6x2 | 0.647 | 78.56 | 63.76 | 63.88 |
| GTA1 + UI-TARS, 6x2 | 0.606 | 75.65 | 62.05 | 62.05 |
| Qwen3 + UI-TARS, 6x2 | 0.630 | 76.72 | 59.27 | 60.40 |
| Three lineages, views 0-3 | 0.594 | 79.19 | 63.69 | 63.82 |
| Three lineages, views 4-7 | 0.691 | 70.52 | 58.38 | 58.63 |

| Outcome | Observed rho | p | Bootstrap mean rho | 99% CI | P(rho >= 0) |
|---|---:|---:|---:|---:|---:|
| pass@12 | -0.326 | 0.0402 | -0.308 | [-0.624, -0.001] | 0.0049 |
| B3 | -0.355 | 0.0245 | -0.313 | [-0.608, 0.008] | 0.0058 |
| M1 | -0.410 | 0.0085 | -0.337 | [-0.634, -0.011] | 0.0036 |

All 10,000 application-group bootstrap replicates are finite after outer-fold-stratified resampling. The first global-bootstrap execution stopped before emitting or writing an L2 result because about 1.6% of replicates omitted an entire held-out fold. Amendment 004 froze fold-stratified resampling before the successful rerun.

The primary pass@12 direction is negative, but `abs(rho)=0.326 < 0.7`. P-L2 fails and L-K3 triggers.

## L3: Mind2Web

Status: `BLOCKED_PREREGISTRATION_GAP`.

- V-only requires TongUI-7B full plus 11 attention-proposed crops. Only the full view exists. Collision-Law v1-v4 are padding, prediction-centered fixed crops, or lower-resolution full views; they are not attention proposals.
- Mixed requires three deployable lineages with four compliant aligned views each. Six deployable lineages have full-view predictions, but none has four compliant views.
- Lineage-only requires 12 eligible aligned full-view models. Eleven aligned models exist and only six satisfy the frozen deployability band.
- No versioned corrected-label release is available.

L-K4 is `NOT_EVALUATED`. No model, attention layer/token, crop heuristic, padding, or duplication was selected to fill the missing pools.

## L4: proposer ablation

E1 uses GTA1 official attention-ranked proposals with layer 20 and target token comma. Across 18,972 N12 candidate regions, 72.51% fully contain the GT bbox and 74.65% contain its center. Coverage declines from 99.94% full-bbox containment at rank 0 to 61.04% at rank 11.

E1 reuses the semantically identical L1 V-only N12 result: B3 60.09%, M1 60.40%, and pass@12 72.80%.

E2 is `UNAVAILABLE` and was not scored:

- Qwen3-VL-8B-Instruct has no released native attention proposal controls.
- UI-TARS-7B-SFT is Qwen2-VL; the released MVP extraction path is modified Qwen2.5-VL and would require unsupported architecture conversion.

The E1/E2 ablation comparison is therefore unavailable, as required by the fail-closed rule.

## Claim boundary

The evidence supports a narrow allocation result on ScreenSpot-Pro: reducing lineage correlation can improve realized accuracy at fixed moderate budgets, and mixed allocation has substantially more oracle headroom. It does not establish a universal correlation law, does not show allocation dominates selector choice at every budget, and does not transfer the claim to Mind2Web. L3 remains blocked rather than negative; L4 remains an unavailable ablation rather than evidence for shared proposals.

Compact source artifacts: `L1_RESULTS.json`, `L2_RESULTS.json`, `L3_STATUS.json`, and `L4_RESULTS.json`.
