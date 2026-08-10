# Learned Structural Aggregator Main Table

状态：`SAFE_BUT_NO_SIGNIFICANT_GAIN`

## Main pooled model

| Benchmark | Oracle pass@12 | CEV-A fallback | LSA direct | LSA safe | Safe − CEV-A | 99% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mind2Web | 57.16% | 32.02% | 28.41% | **33.03%** | +1.01 pp | [−0.81,+2.83] |
| ScreenSpot-Pro | 79.19% | 63.88% | 62.68% | **63.63%** | −0.25 pp | [−0.65,+0.07] |

| Comparison | Mind2Web | 99% CI | ScreenSpot-Pro | 99% CI |
| --- | ---: | ---: | ---: | ---: |
| LSA safe − nested dev-selection | +1.20 pp | [−0.83,+3.18] | −0.19 pp | [−0.73,+0.42] |
| LSA safe − direct | **+4.62 pp** | **[+2.49,+6.75]** | +0.95 pp | [−0.34,+2.30] |
| LSA direct − CEV-A | **−3.61 pp** | **[−6.21,−0.88]** | −1.20 pp | [−2.53,+0.07] |

## Main gates

| Gate | State | Interpretation |
| --- | --- | --- |
| L1 safety vs CEV-A | PASS both | Screen loss 0.25 pp is below MDE 0.70 pp |
| L2 significant useful gain | **FAIL** | Mind2Web +1.01 pp CI crosses zero |
| L3 strong vs dev-selection | FAIL | Neither benchmark has lower CI > 0 |
| L3 balanced safe contribution | FAIL | Standardized balanced 99% CI [−0.88,+2.54] |
| L4 override necessity | PASS | Safe is pointwise better than direct on both |
| LSA-K1/K2/K3/K4 | false | Data, safety, finite threshold, pooled transfer pass |
| LSA-K5 | not applicable | Main model uses no source identity and has no significant gain claim |

Fold choices: H4/H3/H3/H3/H3. Thresholds: 0.0333 / 0.1216 / 0.1666 / 0.1073 / 0.3740. Infinity threshold was never selected.

## Required variants

| Variant | Mind2Web safe | Δ vs CEV-A | ScreenSpot safe | Δ vs CEV-A |
| --- | ---: | ---: | ---: | ---: |
| Reliability only | 32.12% | +0.10 pp | 63.88% | 0.00 pp |
| No geometry | 31.88% | −0.14 pp | 63.88% | 0.00 pp |
| No action | **33.75%** | **+1.73 pp [ +0.19,+3.32 ]** | 63.82% | −0.06 pp [−0.27,+0.19] |
| No parameter | 33.08% | +1.06 pp | 63.63% | −0.25 pp |
| Within-benchmark ranker | 32.02% | 0.00 pp | 63.82% | −0.06 pp |

No-action 相对 nested dev-selection：Mind2Web +1.92 pp [ +0.19,+3.70 ]，ScreenSpot 0.00 pp [−0.50,+0.56]；equal-benchmark standardized CI 为正。但它是预注册消融而非主模型，不升级为确认性主张，需在未用于发现的候选池上确认。
