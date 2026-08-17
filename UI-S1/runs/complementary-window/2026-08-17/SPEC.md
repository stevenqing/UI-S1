# Complementary Window Pilot

Date: 2026-08-17

Status: `PREREGISTERED_DESIGN_ONLY_GPU_NOT_AUTHORIZED`

Evidence status: `POST_SELECTION_EXPLORATORY_PILOT`.

## Boundary

This pilot is motivated by COVER and does not revive X2 or SPLIT. X2 maximized containment through adaptive zoom; this pilot deliberately proposes spatial evidence outside the common 11-crop intersection. SPLIT used two-mode falsification crops and a verifier/flip framing; this pilot adds a proposal directly to the candidate pool and does not flip an existing answer.

ScreenSpot-Pro labels have already been used. No current-data result can be confirmatory.

This document does not authorize GPU inference. A later execution amendment must freeze the model, checkpoint/revision, prompt, crop geometry, sampling, exact call count, expected runtime, and authorization receipt.

## Frozen headroom ledger

COVER measured 1,581 rows:

- common 11-crop target coverage: 931 rows, B3 accuracy 81.95%;
- partial 1-10 coverage: 425 rows, B3 accuracy 57.41%;
- uncovered by all 11 crops: 225 rows, B3 accuracy 0%;
- low-coverage total: 650 rows (41.11%);
- common-minus-low accuracy: +44.42 pp, 99% CI [+34.45,+53.96].

The optimistic reachable-error count is not a predicted gain. A proposal can also damage rows and correlate with the existing pool. Before GPU authorization, a result-free ledger must specify:

$$
\Delta_{net}=P(g)\left[P(\text{rescue}\mid g)-P(\text{harm}\mid g)\right],
$$

where $g$ is a public spatial gate. The ledger must show the rescue rate required to exceed the 0.70 pp MDE at frozen gate prevalence under harm-rate sensitivities `[0,0.05,0.10,0.20]`. It must not use target labels to choose the gate.

## Public spatial gate

The only eligible gate is derived from the 11 public proposer rectangles and existing candidate coordinates. It may use coverage-map geometry, candidate-point coverage counts, crop overlap, or public source disagreement. It may not use target bbox, correctness, `ui_type`, recoverable class, COVER target stratum, or any label-derived statistic.

The gate and threshold must be selected and committed using public fields only before any new model output or label access. If no public gate can target a nontrivial subset without labels, execution stops.

## Complementary window geometry

For each gated row, construct one or more windows by minimizing overlap with the 11 existing crop regions subject to fixed image-bound constraints and a preregistered area. Window count, dimensions, candidate-center rule, tie-break, and clipping must be frozen in the execution amendment.

The new window cannot be selected from multiple post-result geometry families. Random and matched-area controls are mandatory.

## Proposal and aggregation

The new model output enters the existing candidate pool as an additional proposal. No verifier, answer flip, two-mode restriction, target-conditioned crop, or post-hoc acceptance rule is allowed.

Primary aggregation must be chosen before inference from existing canonical rules. Mandatory comparisons include original B3, original nested dev-selection, matched random window, and matched-area high-overlap window.

## Dependence gate

Before accuracy adjudication, report the new proposal's failure phi against every original source on development rows. Continue only when its mean failure phi is below the frozen ScreenSpot-Pro within-lineage reference 0.672 and below the original pool's mean cross-lineage reference 0.577. This is a necessary dependence gate, not sufficient evidence of gain.

## Retention

Every new forward must comply with `docs/generation_trace_retention_policy.md`: token IDs, per-token logprobs or explicit unavailability, coordinate-token spans, raw/normalized sequence scores, decoding parameters, model revision, prompt hash, image hash, and per-shard SHA-256.

## Authorization prerequisites

GPU execution remains prohibited until a committed amendment includes:

1. public gate implementation and prevalence without labels;
2. net-benefit sensitivity ledger;
3. one frozen model and checkpoint;
4. one frozen window geometry family;
5. fixed call budget and resource estimate;
6. controls, aggregators, folds, and kill conditions;
7. protected-process audit and explicit GPU authorization.

Failure of any prerequisite closes the pilot before GPU.