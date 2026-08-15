# XSCR Same-Screen Cross-Row Feasibility Report

Date: 2026-08-14

Outcome: `XSCR_COMPLETE_BELOW_MDE_EXPLORATORY_SPEC_AUTHORIZED`

Evidence status: `POST_SELECTION_FEASIBILITY`. This round is descriptive, is not confirmatory, evaluates no method, and changes no existing project status.

## Structure

| Lane | Rows | Screens | Q1 / median / Q3 | Singleton screens | Rows on singleton screens |
| --- | ---: | ---: | ---: | ---: | ---: |
| androidcontrol_high | 1400 | 1392 | 1.0 / 1.0 / 1.0 | 99.50% | 98.93% |
| androidcontrol_low | 1400 | 1392 | 1.0 / 1.0 / 1.0 | 99.50% | 98.93% |
| mind2web | 1460 | 1402 | 1.0 / 1.0 / 1.0 | 97.50% | 93.63% |

Byte-identical screens are overwhelmingly singletons: 97.50% for Mind2Web and 99.50% for each AndroidControl setting. The public-only seal audit also falsified the assumption that byte-identical Mind2Web screens never cross existing folds; a future transductive evaluation must isolate by screen rather than rely on row folds alone.

## Collision and paired bounds

| Lane | Tolerance | Collision | Repairable | Damageable | Signed proxy | Shared target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| androidcontrol_high | 0.07 | 0.29% | 0 (0.00%) | 1 (0.07%) | -0.071 pp | 40.00% |
| androidcontrol_high | 0.14 | 0.43% | 1 (0.07%) | 1 (0.07%) | +0.000 pp | 40.00% |
| androidcontrol_high | 0.28 | 0.43% | 1 (0.07%) | 1 (0.07%) | +0.000 pp | 40.00% |
| androidcontrol_low | 0.07 | 0.43% | 0 (0.00%) | 0 (0.00%) | +0.000 pp | 40.00% |
| androidcontrol_low | 0.14 | 0.43% | 0 (0.00%) | 0 (0.00%) | +0.000 pp | 40.00% |
| androidcontrol_low | 0.28 | 0.43% | 0 (0.00%) | 0 (0.00%) | +0.000 pp | 40.00% |
| mind2web | 0.00114543 | 3.97% | 0 (0.00%) | 1 (0.07%) | -0.068 pp | 26.88% |
| mind2web | 0.00229087 | 3.90% | 0 (0.00%) | 1 (0.07%) | -0.068 pp | 26.88% |
| mind2web | 0.00458174 | 3.77% | 1 (0.07%) | 2 (0.14%) | -0.068 pp | 26.88% |
| mind2web | 0.250594 | 6.23% | 7 (0.48%) | 1 (0.07%) | +0.411 pp | 82.80% |
| mind2web | 0.501187 | 6.23% | 8 (0.55%) | 1 (0.07%) | +0.479 pp | 97.85% |
| mind2web | 1.00237 | 6.37% | 8 (0.55%) | 2 (0.14%) | +0.411 pp | 100.00% |

AndroidControl's collision surface is at most 0.43%, and its signed screening proxy is never positive. Mind2Web collision ranges from 3.77% to 6.37%. Its best paired structural proxy is **+0.479 pp** at tolerance 0.501187, below the preregistered 0.70 pp MDE. The best AndroidControl proxy is +0.000 pp.

The Mind2Web shared-target diagnostic rises to 97.85% at tolerance 0.501187. Large tolerances therefore merge genuinely shared targets as well as competing locations, supporting soft rather than hard exclusion.

## Decision

The default evidence-based decision would be to close the method direction because the optimistic signed proxy is below MDE and AndroidControl supplies no positive net surface. The recorded human decision instead authorizes writing an **exploratory** soft-assignment specification. That future round remains post-selection and cannot claim confirmation or enter the existing main table as a same-protocol improvement. Correction 004 determined that all private-label files were parsed during input locking, so the nominal 30% subset is not an unread prospective holdout. Any current-data follow-up must use explicitly post-selection nested evaluation; independent validation requires new untouched data.

Q3, Q4, and shared-target diagnostics are evaluation-side only. They do not define a runtime gate.

## Protocol correction

The seal excluded holdout screens from every reported aggregate, but the private-input locker and Q3/Q4 loader read all private-label and reference rows into memory. The holdout is therefore contaminated for future evaluation. See `CORRECTION_004_HOLDOUT_LABEL_ACCESS.md`.
