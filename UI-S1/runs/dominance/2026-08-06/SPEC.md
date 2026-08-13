# Post-B2 Spec: Dominance Law and Freeze

Date: 2026-08-06

Status: frozen execution specification. No new model inference and no new aggregation method are allowed.

Upstream recovery state: B1 passes at both scales, B4 proposer-specific attribution is not supported, B2 fails its cross-scale gate, B-K4 triggers, B3x is cancelled, and the recovery status is `COMPLETE_WITH_RECOVERY_DRIFT`.

The user-specified upstream alias `runs/sourcebias/2026-08-06/` is not present in the workspace. The controlling artifacts are under `runs/sourcebias/2026-08-03/`, updated by the 2026-08-06 recovery run. This path correction does not alter the protocol.

## Scope

This run performs only:

1. D0: audit the degenerate R7 implementation and rerun the frozen nested comparison if needed;
2. D1: test the dominance-gap law from existing traces only;
3. D2: run the frozen cross-benchmark replacement pools when row-level traces are available;
4. freeze the path-B diagnostic-paper claims.

All work is CPU-only. Existing frozen and recovery outputs must remain distinguishable.

## D0: R7 audit

Audit weighted-centroid normalization, candidate/group index alignment, and zero-weight handling. If the implementation is faulty, compare pre-fix and post-fix results and rerun nested selection. Never report only the better result.

Kill condition D-K1 triggers only if the repaired R7 changes the B2 nested conclusion. The required artifact is `d0_r7_audit.json`.

## D1: dominance-gap law

Hypothesis: aggregation gain over the strongest member decreases as the gap between the strongest and second-strongest members increases.

For ScreenSpot-Pro, use the frozen views 0--11 from the three-lineage bank. Enumerate every action-level two-lineage and three-lineage pool with one action per retained lineage. Record:

- strongest and second-strongest member accuracy;
- dominance gap and mean member quality;
- mean pairwise failure kappa;
- frozen B3 and fold-local M1;
- B3/M1 minus best member.

For Mind2Web and AndroidControl, use the frozen T1/T2 pools and add dominance gap to each pool table. Do not infer mixed metrics from marginal summaries.

Report raw Spearman correlation and partial rank correlation controlling mean member quality and mean pairwise failure kappa. Use 10,000 pool-bootstrap replicates, stratified by pool size, seed 20260806.

The law passes only if the combined raw rho is below -0.6, its 99% CI upper bound is negative, the controlled effect remains negative, and every benchmark has the same direction. Missing benchmark rows cause fail-closed non-adjudication, not a positive result.

The required artifacts are `d1_dominance_law.json` and `fig_dominance.pdf`.

## D2: cross-benchmark transfer

Use `runs/final/2026-08-04/configs/t1_t2_pools.yaml` and its frozen runner. Mind2Web holds TongUI-7B fixed across M-cross-3 and M-same-3. AndroidControl reports both same-family brackets and does not attribute a cross-family difference to correlation because member-quality variance is confounded.

If `rows.parquet` and source predictions are missing, record exact missing assets and retain only manifest-level member-quality anchors. No transfer direction, mixed-pool accuracy, failure kappa, or D-K3 decision may be reconstructed from marginal score files.

## Writing decision

The default paper is path B: a diagnostic paper. D1 enters the main text only if its full gate passes. Otherwise the 7B/72B split remains an unexplained limitation. D2 transfer claims require row-level execution.

## Prohibitions

- no new aggregation method;
- no GPU inference;
- no post-result pool changes;
- no best-result selection after R7 repair;
- no cross-benchmark claim from marginal summaries;
- no byte-exact reproduction claim for recovered banks;
- no absolute-score or strongest-zoom head-to-head claim.
