# Final Execution Main Table

Date: 2026-08-04

This is an execution-status table, not paper prose. B2 has not been adjudicated.

| Priority | Work item | Status | Current result or blocker |
|---:|---|---|---|
| 1 | M0 manifest reconciliation | COMPLETE | Five correctness flips: 3 rescues and 2 regressions, net +1/1,581. Two are cross-lineage. Canonical drop-in is +3.605 pp, 99% CI [+1.310,+6.224] pp; CCM-minus-B3 is +0.127 pp. |
| 1 | B1 two-stage source bias | CODE_READY_BLOCKED_FROZEN_BANK | Divisible 72B N9/N12 pools and cluster/representative amplification implemented. |
| 1 | B4 attribution | CODE_READY_BLOCKED_FROZEN_BANK | Deterministic sensitivity retained; 10,000-draw global action-subset balancing implemented. |
| 1 | T1/T2 cross-benchmark replacement | CONFIG_AND_RUNNER_READY_BLOCKED_ROWS | Pools frozen and exact quality-confound anchors verified; `rows.parquet` is absent. |
| parallel | S0 SafeGround anchor | COMPLETE_ALGORITHM_LEVEL_PORT | Official GTA1-7B U_COM AUROC 0.6344; local K4/T0.7 result 0.6278 does not pass the table-precision anchor and protocol differs from K10/T1.0. The official repository ships no K10 prediction artifact for zero-GPU replay. |
| 2 | B2 combined 24 methods | CODE_READY_BLOCKED_FROZEN_BANK | Combined-24 and R0-only nested selectors implemented; paper-shape rule frozen before results. |
| 3 | B3x reclaim | GATED | Runner implemented; cannot execute before combined B2 passes. |
| parallel | X1 GTA1 sampling N16 | COMPLETE_REUSED_TRACE | 1,581 x 16 existing trace reused. S-only GUI-RC slope -0.000285, 99% CI [-0.000789, +0.000203]; title scope is fixed-view allocation axis. |

## S0 positioning

| Quantity | Value |
|---|---:|
| Official SafeGround GTA1-7B U_COM AUROC | 0.6344 |
| Local stochastic GTA1 K4/T0.7 GUI-RC AUROC | 0.6278 |
| V-only N12 transferred correctness AUROC | 0.7442 |
| Mixed N12 transferred correctness AUROC | 0.8297 |
| Mixed minus V-only B3 at 80% retained coverage | +7.12 pp |

The deterministic N12 result transfers only the dispersion score. It does not inherit SafeGround's K=10 stochastic protocol, Learn-Then-Test calibration, Clopper-Pearson bound, or FDR guarantee.

## Required frozen assets

See `ASSET_PREFLIGHT.json`. The missing groups are:

- 7B H1/H3/Allocation candidate banks for M0, B1/B4 and B2;
- 72B G2 region/score traces plus shared labels for B1/B4 and B2;
- `runs/complementarity/2026-07-30/rows.parquet` for T1/T2.

These are Git-ignored traces from the source machine. They must be copied exactly rather than reconstructed from aggregate summaries.
