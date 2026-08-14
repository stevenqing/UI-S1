# SPLIT Falsification-Crop Probe Report

Date: 2026-08-14

Status: `SPLIT_STOPPED_PRE_GPU_Z_K6_GEOMETRY_AND_Z_K7_LOW_N`

## Scope

SPLIT was preregistered as an exploratory pilot. It does not change F1, Q1, `TRIVUS_NOT_PROMOTED`, or VUS-SR, and it does not authorize a deployable method. No model forward or GPU authorization occurred.

## Zero-GPU gate

Z-G1 passed. Across 1,581 held-out ScreenSpot-Pro rows, nested selection chose $g=0.25$ in all five folds. The gate triggered on 1,187 rows (75.08%). There were 102 M2-only positive rows, giving pooled $\Delta_2=6.45$ pp and an 8.59% conditional positive rate inside the gate.

This is candidate-level headroom only. It does not show that falsification-crop confidence can identify the M2-only rows.

## Geometry stop

The corrected matched-window audit retained 869/1,187 gate rows and failed 318/1,187, a 26.79% failure rate above the preregistered 15% Z-K6 limit. Exactly 163 rows failed only `W1_excludes_M2`; 155 failed only `W2_excludes_M1`. All $W_0$ exclusion checks, area/aspect matching, and Qwen3/GTA1 resize equality checks passed.

The failures occur when image boundaries prevent the fixed minimum 512-pixel window from extending away from the neighboring mode. The frozen protocol prohibits shrinking the window, changing the separation axis, or rescuing failed rows. Z-K6 therefore stops the round before GPU.

After geometry, only 76 positive rows remain, below the preregistered minimum 120. Z-K7 independently limits any continuation to an observational report.

## Endpoints and conclusion

Z-P3, Z-P1, Z-P2, Z-P4, and Z-P5 are all `NOT_RUN_PRE_GPU_STOP`. Qwen3 and GTA1 forward counts are zero; Qwen2.5 remains deferred because its checkpoint is absent. No balanced subset or GPU authorization was created.

The strongest defensible conclusion is:

> A two-mode candidate headroom of 6.45 pp exists under the frozen gate, but the preregistered falsification-crop geometry is infeasible often enough to trigger Z-K6, and the surviving positive set is too small for endpoint decisions. SPLIT provides no evidence that crop confidence is an orthogonal channel.

Any alternate crop geometry is a new study and cannot rescue SPLIT post hoc.
