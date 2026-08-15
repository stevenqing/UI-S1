# DECOMP Pool Allocation, Same-Screen Structure, and Logprob Report

Date: 2026-08-15

Outcome: `DECOMP_COMPLETE_LINEAGE_AXIS_DOMINANT_LOW_SCREEN_STRUCTURE_NO_LOGPROB`

DECOMP is a zero-GPU recomputation over frozen artifacts. It changes no existing result status and introduces no method claim.

## P0 reconciliation

P0 passed after correcting three apparent source issues. XSCR's 1,460/1,400 values are post-seal exploratory row counts, not full-bank sizes. SPLIT's Qwen2.5 name referred to a deferred probe, not a C-uni lineage. The historical 2,094-action Mind2Web DOM lane remains locally unavailable. Arm 1 is therefore ScreenSpot-Pro-only. Full details and hashes are in `LANE_RECONCILIATION.md`.

## Arm 1: allocation decomposition

The full-pool density B3 anchor reproduces at **63.69%** over 1,581 rows. Arm 1 evaluates 4,083 subsets with budgets 2-12; the 12 singleton subsets are outside the requested budget range. Subsets overlap, so uncertainty resamples application groups/rows and never subsets.

Baseline accuracies are: full-pool majority/best-single **59.84%**, full-pool density B3 **63.69%**, A2/A3 **63.88%**, A4 **63.95%**, and nested dev-selection **63.82%**.

The lineage marginal is positive at every identifiable budget through B=8: density B3 ranges from **+0.73** to **+1.47 pp**, and F1 majority from **+0.24** to **+1.06 pp**. View marginals are much smaller and are usually negative beyond the smallest budgets. This direction is consistent with the historical failure kappas, view $\kappa=0.895$ versus cross-lineage $\kappa=0.398$.

The largest lineage variance shares occur for density B3 around intermediate budgets, while view and interaction shares are generally smaller. B=12 has only one configuration, so its ANOVA and marginal contrasts are `NA`. Every selected budget cell touches at least one supported-axis boundary in every fold; the budget table is therefore a boundary-sensitive descriptive recommendation, not a stable optimizer or new method.

Mind2Web remains `BLOCKED_ALIGNED_POOL_UNAVAILABLE` and has no Arm 1 table.

The complete budget and mechanism tables are in `MAIN_TABLES.md`.

## Arm 2: label-free same-screen structure

| Grouping | Rows | Screens | Q1 / median / Q3 | Singleton screens | Rows on singleton screens |
| --- | ---: | ---: | ---: | ---: | ---: |
| image_sha256 | 1581 | 1551 | 1.0 / 1.0 / 1.0 | 98.52% | 96.65% |
| img_filename | 1581 | 1581 | 1.0 / 1.0 / 1.0 | 100.00% | 100.00% |

| Tolerance | Collision rows | Collision screens |
| ---: | ---: | ---: |
| 7 px | 2/1581 (0.127%) | 1/1551 (0.064%) |
| 14 px | 0/1581 (0.000%) | 0/1551 (0.000%) |
| 28 px | 4/1581 (0.253%) | 2/1551 (0.129%) |

Byte hashing reveals repeated screenshots hidden behind distinct source filenames: source IDs are all singletons, while byte hashes yield 1,551 screens. Even under byte identity, 98.52% of screens are singletons and 96.65% of rows have no same-screen partner. Collision rates are 0-0.253% across `[7,14,28]` pixels. No label, target bbox, evaluator, or prohibited path was opened.

The evidence-based default is to close this lane. The recorded human decision instead authorizes writing a physically isolated label-process specification. That specification is not authorized for execution and cannot restore confirmation because ScreenSpot-Pro labels were already used. See `runs/xscr-label-isolated/2026-08-15/SPEC.md`.

## Arm 3: logprob inventory

Arm 3 inspected 16 ScreenSpot-Pro generating-trace files (4,743 rows across files) and 44 Mind2Web files (20,800 rows across files). It found zero files with generating-model logprobs, generated token IDs, or coordinate-token spans. No labels were opened and no AUROC was computed.

Downstream selector logits exist but are explicitly classified as `DOWNSTREAM_CANDIDATE_SELECTOR_NOT_GENERATING_MODEL_LOGPROB`; they were not substituted. The arm stops as `LOGPROB_CHANNEL_NOT_RETAINED`.

Future forward retention requirements are now repository policy in `docs/generation_trace_retention_policy.md`.

## Boundaries

Arm 1 is a post-hoc descriptive decomposition of the existing +3.605 pp ScreenSpot-Pro pool result. Arm 2 is label-free structure only. Arm 3 is mechanism inventory only. None changes F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, or XSOFT.
