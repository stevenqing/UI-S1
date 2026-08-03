# CALA: Complementarity-Aware Lineage Allocation

Date: 2026-08-03

Status: result-blind method preregistration. No CALA allocation, held-out accuracy, or 72B equal-budget result existed when this document was written. The completed Allocation-Law and Scale-Up evidence motivates the method but does not determine any CALA action sequence.

## 1. Motivation

Existing GUI grounding test-time scaling mainly spends additional forwards on more views from one model. Our controlled 7B result shows that reallocating the same 12-forward budget from correlated GTA1 views to shared views across GTA1, Qwen3-VL, and UI-TARS improves unchanged B3 and fold-local M1. Static `4 x 3` allocation, however, is still a manually specified ensemble.

CALA turns the observation into a budgeted allocation algorithm. Its action space is the Cartesian product of model lineage and shared proposal view:

`u = (model, view)`.

Given a fixed forward budget, CALA selects actions that maximize held-out marginal candidate coverage rather than repeatedly selecting individually strong but failure-correlated units.

## 2. Candidate bank and isolation

The primary 7B bank has 36 actions: three frozen models by shared GTA1 proposal views 0-11. Every action already has one prediction on each of 1,581 ScreenSpot-Pro identities. Existing inference is treated as a counterfactual full-information bank; evaluation masks every action not selected by the policy.

The same frozen 26 application groups and load-balanced five folds are used. For outer fold `f`, allocation fitting and router training use only the other four folds. Test labels, correctness indicators, unselected test predictions, and aggregate test results are never visible to the allocator.

At simulated deployment time, a policy sees only input metadata, shared proposal metadata, and predictions from actions it has already paid for. Accessing an unselected point for routing is a hard failure.

## 3. CALA-S: static complementarity allocation

CALA-S is the parameter-free core method. For each outer fold it constructs one nested 16-action sequence greedily on development rows.

For selected set `S`, define empirical candidate coverage:

`C(S) = mean_i any(action u in S is correct on development row i)`.

At each step choose the unselected action with maximum marginal coverage:

`Delta(u | S) = C(S union {u}) - C(S)`.

Ties are resolved in this frozen order:

1. higher development B3 accuracy after adding the action;
2. lower mean pairwise failure kappa within the resulting set;
3. higher individual development accuracy;
4. model order GTA1, Qwen3-VL, UI-TARS, then lower view index.

The N=4/8/12/16 CALA-S pools are exact prefixes of the learned sequence. The test fold executes only its selected prefix. B3 is unchanged and primary; fold-local M1 and pass@N are secondary.

Coverage is monotone submodular, so greedy selection directly rewards novel correct candidates and penalizes redundant successes without a tuned diversity coefficient.

## 4. CALA-A: adaptive scout-and-top-up allocation

CALA-A is the instance-adaptive extension. It is evaluated at N=8/12/16.

### Scout

Every row first executes six fixed actions:

- each of the three models on view 0;
- each of the three models on view 1.

The scout is identical across methods and costs six forwards.

### Router target

For an unselected action `u` and observed history `S`, the development target is:

`novel_correct(u, S) = correct(u) and no selected action in S is correct`.

The router estimates the probability that the next action adds a previously missing correct candidate. It uses logistic regression with L2 regularization `C=1`, standardized continuous features, balanced class weights, and seed 20260803. No hyperparameter search is allowed.

### Router features

Only pre-action features are permitted:

- action model one-hot and normalized view index;
- proposal rank, normalized crop area and center, and GTA coverage for that shared view;
- development-only individual action accuracy;
- development-only mean failure kappa between the action and selected units;
- selected count per model;
- normalized centroid, x/y spread, mean pairwise distance, maximum pairwise distance, and largest 14-pixel cluster fraction of observed selected points;
- image aspect ratio, log pixel area, and instruction length.

The unselected action's point, parse outcome, confidence, or correctness is prohibited.

### Training states

For each development row, create four deterministic trajectories. Each starts from the six scout actions and appends a seed-derived random permutation of remaining actions. States after 0 through 9 appended actions contribute training examples for every still-unselected action. Seeds are SHA-256 derived from base seed 20260803, row identity, outer fold, and trajectory index.

At test time, CALA-A repeatedly selects the action with maximum predicted novel-correct probability, reveals only that action's stored prediction, updates history features, and continues to the budget. Ties use higher development individual accuracy, then frozen action order.

## 5. Baselines

All baselines use identical candidate geometry and exact budgets.

- `V-only`: GTA1 views in official rank order.
- `Uniform-Mixed`: existing round-robin sequence by view then model.
- `Quality-Only`: development individual accuracy descending, then frozen action order.
- `Random`: one nested action permutation per outer fold, seed 20260803.
- `CALA-S`.
- `CALA-A` for N=8/12/16.
- `Oracle-Coverage`: row-level label-aware greedy allocation, shown only as an unattainable upper bound.

## 6. Evaluation and statistics

Primary method/rule: N12 B3.

Primary comparison: CALA-S N12 B3 minus Uniform-Mixed N12 B3.

Secondary comparisons:

- CALA-A minus Uniform-Mixed and CALA-S at N=8/12/16;
- M1 and pass@N versions of all comparisons;
- accuracy-cost curves;
- mean pairwise failure kappa;
- action frequency by model/view and application group;
- weak-lineage ablation excluding UI-TARS;
- same-model pseudo-lineage control using GTA1 views only.

Use 10,000 paired application-group bootstrap replicates stratified by frozen outer fold, seed 20260803, 99% percentile intervals, and plus-one one-sided p-values.

The frozen ScreenSpot MDE is 0.007043345177520599.

Primary success requires all of:

- CALA-S N12 B3 delta over Uniform-Mixed is positive;
- 99% CI lower bound is positive;
- delta exceeds the frozen MDE;
- CALA-S pass@12 is not lower than Uniform-Mixed pass@12.

Adaptive success is adjudicated separately and requires CALA-A N12 B3 to exceed CALA-S with a positive 99% CI lower bound. Failure of CALA-A does not permit redesign or invalidate CALA-S.

## 7. 72B equal-budget transfer

The Scale-Up G2 absolute experiment remains unchanged. CALA adds a separate N8 equal-budget evaluation after the existing label-free score traces are complete.

The guaranteed 72B action universe is three models by shared regions 0-3. Compare:

- GTA1-only N8 fallback;
- Uniform-Mixed N8, first eight view-major/model-minor actions;
- fold-local CALA-S N8;
- CALA-A N8 with the same six-action scout and two adaptive top-up actions.

All configurations use exactly eight scored model-region forwards per row. The 72B analysis is cross-fitted by application groups and never changes the completed G2 N12 absolute result. No same-compute claim is made from G2 P2 N12 versus P1 N8.

## 8. Claim boundaries

- The novelty is budgeted lineage-view allocation, not model ensembling itself.
- No claim of being the first multi-model GUI ensemble is made without a separate literature audit.
- The allocator may use development labels but is label-free on held-out rows.
- B3 remains unchanged in the primary result, isolating allocation from selector changes.
- A manually chosen pool cannot be relabeled as CALA.
- Scale-Up raw traces, checkpoints, and logs remain untracked.

## 9. Deliverables

- `configs/protocol.yaml`
- `cala_common.py`
- `cala_static.py`
- `cala_adaptive.py`
- `cala_evaluate.py`
- `cala_results.json`
- `MAIN_TABLE.md`
- `REPORT.md`
