# Scale-Up Main Table

## Controlled 7B paired results

| Comparison | Left | Right | Delta | 99% CI | One-sided p |
|---|---:|---:|---:|---:|---:|
| Mixed N12 B3 H3-native vs V-only N12 B3 | 63.63% | 60.09% | +3.54 pp | [+1.27, +6.15] pp | 9.999e-05 |
| Mixed N12 M1 vs GTA1-7B bare | 63.82% | 49.40% | +14.42 pp | [+10.39, +18.84] pp | 9.999e-05 |
| Mixed N12 M1 vs Qwen3-VL-8B-Instruct bare | 63.82% | 54.65% | +9.17 pp | [+5.71, +13.01] pp | 9.999e-05 |
| Mixed N12 M1 vs UI-TARS-7B-SFT bare | 63.82% | 33.46% | +30.36 pp | [+24.47, +36.73] pp | 9.999e-05 |
| Mixed N12 M1 vs V-only N12 M1 | 63.82% | 60.40% | +3.42 pp | [+1.41, +5.67] pp | 9.999e-05 |
| Mixed N16 B3_mvp vs V-only N16 B3_mvp | 63.76% | 58.32% | +5.44 pp | [+2.86, +8.41] pp | 9.999e-05 |
| Mixed N16 M1_ccm vs V-only N16 M1_ccm | 63.76% | 58.25% | +5.50 pp | [+3.09, +8.16] pp | 9.999e-05 |

The H3-native B3 comparison (63.63% vs 60.09%) is primary. The later Allocation/Closing reconstruction gives 63.69% for Mixed B3 and is retained only as an implementation-sensitivity check.

## G1 72B lineage gate

| Model | Local bare | Paper-only reference | Difference | Anchor within 2 pp |
|---|---:|---:|---:|---|
| GTA1-72B | 59.01% | 58.40% | +0.61 pp | True |
| Qwen3.5-122B-A10B | 70.84% | 70.40% | +0.44 pp | True |
| UI-Venus-Ground-72B | 60.40% | 61.90% | -1.50 pp | True |

| Pair | Failure kappa | Matched-marginal p |
|---|---:|---:|
| GTA1-72B__Qwen3.5-122B-A10B | 0.460 | 0.000999 |
| UI-Venus-Ground-72B__GTA1-72B | 0.677 | 0.000999 |
| UI-Venus-Ground-72B__Qwen3.5-122B-A10B | 0.464 | 0.000999 |

pass@3 is 80.20%. G1 pass is `False`; action is `RUN_G2_MARGINAL_GATE_STANDARD_THRESHOLD`.

## G2 controlled 72B pools

| Pool | Budget | B3 | M1 | pass@N |
|---|---:|---:|---:|---:|
| P1 GTA1-72B single lineage | 8 | 23.85% | 25.74% | 69.32% |
| P2 mixed 72B | 12 | 32.83% | 49.15% | 84.50% |

P2-P1 M1 is +23.40 pp, reported only as unequal-budget context (P2 N12 versus P1 N8). Proposal MDE is +2.35 pp. Outcome: `BELOW_PAPER_MODEL_REFERENCE`.

## Paper-only context

| System/model | ScreenSpot-Pro | Comparability |
|---|---:|---|
| Qwen3.5-122B-A10B reported model | 70.40% | Paper only; excluded from paired calculations |
| ZoomClick + UI-Venus-Ground-72B | 73.10% | Paper only; excluded from paired calculations |
| MVP trained GRPO selector | 62.80% | Paper only; excluded from paired calculations |
