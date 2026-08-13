# Final Execution Report

Date: 2026-08-04

## Completed

The execution-order specification and pre-result paper-shape decision are frozen. Path A requires the combined 24-method B2 nested result to pass the original two-scale MDE/99% CI gate and reach at least 67% at 7B. Any failed requirement selects Path B. No B2 result existed when this rule was written.

S0 is complete. The official SafeGround paper reports GTA1-7B U_COM AUROC 0.6344 under K=10, temperature 1.0, patch size 14 and beta 0.3. The available local stochastic trace is K=4, temperature 0.7 and produces 0.6278 with the commit-5e8fca7 geometry port. It does not pass a comparable numerical anchor and is retained only as an algorithm-level transfer. The official commit contains source code and figures but no K10 GTA1 prediction artifact for zero-GPU replay. R4 is narrowed to the supported statement that cross-lineage candidates strengthen this transferred signal: correctness AUROC rises from 0.7442 to 0.8297, and at 80% retained coverage Mixed B3 exceeds V-only B3 by 7.12 percentage points. No local FDR guarantee is claimed.

X1 is also complete without new GPU inference. The existing 1,581-row, 16-sample GTA1 trace covers N=4/8/12/16. The S-only GUI-RC slope is -0.000285 per forward with 99% CI [-0.000789, +0.000203]; the B3 slope is approximately zero. This does not establish a negative single-model sampling axis, so the fixed scope is `fixed-view allocation axis`.

## Implemented, awaiting frozen inputs

M0 now compares the same 12 candidates in historical H3 model-major order and canonical L1/CALA view-major order, records both selected candidates and lineages, and recomputes the canonical grouped bootstrap. The committed summaries already prove that exactly one correctness row differs, but they do not retain the row identity or candidate bank. M0 therefore fails closed until the frozen bank is restored.

The amended source-bias implementation now includes 72B N9/N12 equal-slot controls, two-stage cluster-formation and within-cluster representation amplification, and seeded random global subset balancing. B2 now runs the frozen 24-method combined selector and a separate R0-only nested selector. The prior weighted-centroid helper returned an unnormalized weighted sum; it is corrected to divide by total weight and covered by a non-unit-weight regression test. New B2 results must therefore be recomputed rather than borrowing the old R7 grid values. B3x is implemented and refuses to execute unless combined B2 passes.

T1/T2 pools are frozen before row restoration. Mind2Web compares TongUI-7B/CogAgent/UI-TARS-7B against TongUI-3B/7B/32B; both pools share TongUI-7B as strongest member. AndroidControl reports cross-family UI-AGILE-7B/GUI-R1-7B against both Agile-family and GUI-R1-family controls. The cross-family comparison remains explicitly confounded by member-quality variance and cannot be attributed to correlation alone.

## Blocker

The new clone lacks Git-ignored frozen traces. Exact paths are in `ASSET_PREFLIGHT.json`. Re-running model inference would violate the zero-GPU execution classification and would not reproduce source hashes. The correct next operation is an exact copy from the source workspace, after which execution resumes in this order:

1. `m0_manifest_diff.py`
2. amended B1 and B4
3. `t1_t2_transfer.py`
4. combined-24 B2 and paper-shape adjudication
5. `b3x_reclaim.py` only if B2 passes
