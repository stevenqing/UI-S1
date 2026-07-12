# Pass@8 Selector-to-Training Bridge Diagnostic

Positive student-relative utility is not equivalent to clean SFT labels: on student-wrong rows, a wrong replacement has utility zero but remains an actively wrong training target.

## Candidate training variants

| variant | changed rows | correct labels | purity | Wilson 95% | rescue / regress | utility |
|---|---:|---:|---:|---:|---:|---:|
| all_9b_changes | 425 | 46 | 10.82% | [8.21%, 14.14%] | 46 / 1 | +10.59pp |
| all_consensus_changes | 334 | 39 | 11.68% | [8.66%, 15.56%] | 39 / 1 | +11.38pp |
| 9b_consensus_same_action | 114 | 13 | 11.40% | [6.79%, 18.54%] | 13 / 0 | +11.40pp |

## Qwen3.5-9B self-selection

Among all changed 9B selections, 197 / 425 exact selected actions contain a Qwen3.5 source occurrence (46.35%).

| selected exact-source stratum | rows | correct | purity |
|---|---:|---:|---:|
| qwen35_only | 143 | 10 | 6.99% |
| qwen35_mixed | 54 | 10 | 18.52% |
| no_qwen35 | 228 | 26 | 11.40% |

## Safety boundary

The locked population contains only 3 student-correct rows out of 708. Its rescue/regress ratio therefore cannot establish arbitrary-state regression safety.

## Decision

Do not train directly from these selected changes. First measure a controlled 100/80/60/40% purity-response curve at fixed 25/75 revision-to-clean replay, then measure train-split aggregate purity for frozen GT-free construction variants. A training variant is eligible only if its conservative purity bound clears the empirically tolerated purity threshold.
