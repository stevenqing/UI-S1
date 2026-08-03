# CALA Method Report

Date: 2026-08-03

## Method

CALA treats a model-lineage/view pair as a budgeted action. CALA-S greedily maximizes development-only marginal candidate coverage. CALA-A spends six fixed scout forwards, then routes additional actions using a cross-fitted logistic predictor of novel-correct probability. Held-out routing can see only proposal metadata and predictions from actions already executed. Unchanged B3 is primary.

This is more specific than model ensembling: CALA is a fixed-budget action scheduler over model lineage and shared proposal geometry.

## Primary result

The preregistered CALA-S N12 primary failed. B3 changes from 63.69% to 62.18%: -1.52 pp, 99% CI [-3.12, -0.06] pp. CALA-S raises pass@12 from 79.19% to 80.01%, but B3 cannot realize the extra oracle headroom.

The preregistered CALA-A N12-over-CALA-S adaptive success criterion also failed: +0.95 pp, 99% CI [-0.75, +2.36] pp. Against Uniform N12, CALA-A is -0.57 pp.

## Budget-specific positive result

At 7B N8, CALA-A improves unchanged B3 from 61.99% to 63.06%: +1.08 pp, 99% CI [+0.29, +2.10] pp, p=0.0004. pass@8 changes by -0.13 pp with a CI crossing zero.

The gain is budget-specific. Continuing the same router to N12/N16 does not improve Uniform Mixed. The method therefore supports adaptive early allocation and stopping, not monotonic gains from adding routed actions.

## 72B equal-budget transfer

All 72B policies use exactly eight scored model-region forwards. CALA-S improves B3 from 41.24% to 45.41%: +4.17 pp, 99% CI [+1.56, +7.06] pp. Its M1 delta is +3.16 pp, and pass@8 delta is +0.95 pp; both 99% CI lower bounds are positive.

CALA-A also improves 72B B3 over Uniform N8 by +2.72 pp, 99% CI [+0.90, +4.65] pp, but is below CALA-S by -1.45 pp with a CI crossing zero.

This transfer validates the allocation algorithm direction at equal budget, but absolute 72B accuracy remains low and does not rescue the failed Scale-Up SOTA target.

## Contribution

The defensible method contribution is:

> A cross-fitted, complementarity-aware scheduler over model-lineage and shared-view actions improves unchanged GUI grounding aggregation at a fixed low inference budget, with significant N8 gains at both 7B and 72B scales.

Supporting methodological findings:

- candidate coverage is submodular and easy to increase, but coverage-only allocation can hurt the downstream aggregator;
- instance-adaptive routing is beneficial at the first two top-up decisions on 7B;
- static development complementarity transfers strongly at 72B;
- neither policy is universally better across budgets, so the result is an allocation-and-stopping method rather than a generic ensemble law.

## Boundaries

- The N12 primary method claim failed and remains visible.
- No post-result hyperparameter search was performed.
- B3 was not retuned.
- 72B N8 is an equal-forward transfer, separate from the failed N12 absolute-SOTA experiment.
- CALA does not claim that multi-model ensembling itself is novel.
