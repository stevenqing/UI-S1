# Amendment 018: One-Stage Collapse

Date: 2026-08-13

Timing: after complete exploratory publication of 60 cheap and 60 verifier OOF jobs. All labels are open; this is a post-hoc method decision, not confirmation or promotion.

## Result

The full-candidate cheap ranker substantially improves the frozen blind ordering. Candidate AUROC is 0.831 on Mind2Web, 0.844 on ScreenSpot-Pro, and 0.813 on AndroidControl. Top-1 gains over blind ordering are 4.82, 19.06, and 4.56 percentage points respectively.

The second verifier receives the same 115 public dimensions plus five cheap-OOF statistics. It adds only +0.012 points top-1 on Mind2Web, -0.174 on ScreenSpot-Pro, and -0.006 on AndroidControl; MRR and Brier also do not improve consistently.

Cross-fitted stopping calibration selects budget one for every non-fallback cheap policy. Its equal-cell gain over strongest is +0.565 points on Mind2Web and approximately zero on ScreenSpot-Pro and AndroidControl. The verifier stopping policy is weaker.

## Decision

The supported compact candidate method is one benchmark-specific contextual full-candidate ranker followed by a strongest-safe accept/fallback gate. The same-information second Transformer and multi-candidate cascade are removed from the main method.

A genuinely sequential verifier remains a valid future direction only if it introduces a new information channel, such as candidate-specific screenshot/task/action verification from a VLM. Reordering or recalibrating the same frozen features is not a strong verifier.

## Claim boundary

The OOF contexts repeat each sample four times across outer development scopes. Results are exploratory, have no external confirmation, and cannot change the prior `TRIVUS_NOT_PROMOTED` outcome.