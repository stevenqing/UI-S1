# Amendment 001: 72B Equal-Budget Transfer Operations

Date: 2026-08-03

Status: frozen after Scale-Up score-trace completion and before any CALA 72B N8 allocation or accuracy result.

## Available action bank

The three Scale-Up models have complete scored predictions for shared region indices 0-3 on all 1,581 identities. GTA1-72B additionally has complete indices 0-7. No new inference is required.

The mixed action universe is 12 units in view-major, model-minor order:

- GTA1-72B, UI-Venus-Ground-72B and Qwen3.5-122B-A10B on each of regions 0-3.

## Exact N8 pools

- `GTA1_N8`: GTA1 region indices 0-7.
- `Uniform_Mixed_N8`: first eight units in view-major, model-minor order.
- `CALA_S_N8`: one development-only greedy coverage sequence per outer fold, truncated to eight.
- `CALA_A_N8`: six fixed scout units (all three models on regions 0-1) plus two routed top-up units.

Every evaluated row contains exactly eight distinct scored model-region forwards. Region proposal computation is common infrastructure and is not counted differently between pools.

## Static allocation

CALA-S uses the same development marginal pass-coverage objective and tie breaks as the 7B method. Only the action universe changes from 36 to 12. Held-out labels do not enter the sequence.

## Adaptive allocation

CALA-A uses the same feature families and fixed logistic regression as the 7B method. Because N8 deployment has exactly two top-up decisions, development trajectories contribute states only before top-up 1 and before top-up 2. Four deterministic random trajectories are retained per development row.

The router may use only selected prediction coordinates. Proposal metadata for unselected actions remains available because all actions on the same view share the already computed region. Unselected model outputs and correctness are prohibited.

## Evaluation

Primary reporting rule is unchanged B3. Fold-local M1 and pass@8 are secondary. Report 10,000 fold-stratified application-group paired bootstrap replicates, seed 20260803, 99% intervals and plus-one one-sided p-values for:

- CALA-S minus Uniform Mixed;
- CALA-A minus Uniform Mixed;
- CALA-A minus CALA-S;
- all mixed policies minus GTA1 N8.

This transfer tests equal-budget direction and does not alter the failed Scale-Up N12 absolute-SOTA adjudication. No threshold, feature, or action universe may be changed after seeing N8 results.