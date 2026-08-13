# 72B Recovery Analysis

Date: 2026-08-06

## Scope

The GTA1-72B, UI-Venus-Ground-72B, and Qwen3.5-122B-A10B score banks were regenerated to 1,581 rows each and passed their scoring gates. The mixed Scale-Up result and B1/B4/B2 analyses were recomputed from those recovered banks. Historical frozen B1/B4/B2 artifacts remain unchanged; recovery analyses use separate files.

## Reproducibility boundary

The recovered score banks are structurally complete and internally hash-consistent, but they are not byte-identical to the historical frozen banks. The recovered 72B B1 incorrect-row anchor is:

- winning-set members GTA/Venus/Qwen3.5: 1370/1003/369, versus frozen 1374/1000/370;
- final winners GTA/Venus/Qwen3.5: 871/53/5, versus frozen 872/52/5.

The B2 72B B3 baseline is unchanged at 41.24%. Fold-local M1 changes from the frozen 52.12% to 53.19%, a +1.08 pp recovery drift. Recovery outputs explicitly record these mismatches and required opt-in flags; default frozen execution still rejects them.

## Scale-Up result

The mixed N12 pool reaches 84.63% pass@N, 49.15% M1, and 32.45% B3. The high candidate oracle coverage is therefore not converted into a competitive final prediction by the original aggregators. The effective 73.1% threshold and system-SOTA gate both fail.

## B1 source bias

B1 passes at both scales. On 72B B3 incorrect rows, GTA1 wins 871 of 929 rows. Its standardized residual is +35.42 with p approximately 1.21e-273 and Cramer's V 0.822. The source-selection bias conclusion is robust to the small recovery drift.

## B4 attribution

The stronger proposer-specific mechanism is not supported at both scales. The defensible interpretation remains a heterogeneous-pool aggregation effect. Deterministic 72B N8 count balancing raises B3 from 41.24% to 49.72% (+8.48 pp), but the 10,000-draw random global-subset distribution is wide: mean 41.03%, 99% interval [23.34%, 55.22%]. GTA winner overrepresentation remains strong after balancing.

## B2 lineage normalization

At 72B, combined-24 nested lineage normalization reaches 70.52%. It improves over B3 by +29.29 pp (99% CI [+21.21, +35.88]) and over recovered M1 by +17.33 pp (99% CI [+11.96, +21.99]). It remains 0.89 pp below the reported Qwen3.5 best-single result and 0.76 pp below the matched recovered view-0 bank.

At 7B, nested lineage normalization is 63.69%, equal to B3. The preregistered primary criterion requires success at both scales and therefore fails. B-K4 triggers and B3x is cancelled by protocol.

## Conclusion

The recovery supports the mechanism-level result: model lineages provide substantial complementary candidate coverage, while source-sensitive flat aggregation leaves much of that coverage unrealized. Lineage normalization strongly corrects the 72B failure but does not generalize to 7B or exceed the best single 72B model. This is recovery evidence, not byte-exact reproduction of the frozen run.