# TILE Amendment 002: V-only versus C-uni baseline reconciliation

Date: 2026-08-18

Status: `FROZEN_AFTER_FAILED_PREFLIGHT_BEFORE_ANY_TILE_STATISTIC`

The first TILE preflight stopped before writing output because Amendment 001 incorrectly required row-wise B3 equality between COVER and CWIN. No TILE eccentricity, curve, score, ledger, or gate was computed.

## The mismatch

COVER `b3_correct` is canonical C-uni B3 over three lineages by four views: 1,007/1,581 = 63.69%.

CWIN `original_b3_correct` is GTA1 V-only N12 over one full image plus 11 GTA1 crops: 950/1,581 = 60.09%.

The folds agree exactly, but the two B3 outputs disagree on 143 rows. They are different candidate pools and must not be asserted equal.

## Primary TILE baseline

Arm T is one GTA1 full-image forward plus N GTA1 tile forwards. C2 is frozen GTA1 view 0 plus GTA1 views 1 through N. The strict source-matched baseline is therefore frozen GTA1 V-only N12, not C-uni.

Effective immediately:

- Stage-0 y, expected repair/damage/net, N selection, T-G1, and original-correct primary domain use CWIN `original_b3_correct`, 950 rows;
- T-P1 compares Arm T against GTA1 V-only N12 canonical B3;
- T-P2 compares against GTA1 V-only N12 fold-local M1_ccm;
- T-P3 remains the strict equal-budget GTA1 prefix control.

## Mandatory C-uni contextual ledger

C-uni remains the main-table context and must be co-reported, never discarded:

- report Arm-T expected/observed contrasts against C-uni B3 separately as `contextual_C_uni_B3`;
- report C-uni original-correct 1,007-row damage separately;
- report V-only original-correct 950-row damage separately;
- report crop-covered 1,356-row damage separately.

These domains overlap and cannot substitute for one another. The original user-specified 1,007 denominator is retained as the C-uni contextual damage domain, not relabeled as V-only correctness.

OWIN's factorized G_N used C-uni B3 calibration and remains contextual/prohibited as a TILE target. It does not define the V-only primary expected gain.

## Preflight correction

Preflight must require:

- exact row-ID and fold equality across COVER/CWIN/recomputed folds;
- exact C-uni count 1,007;
- exact V-only count 950;
- exact crop-covered count 1,356;
- explicit 143-row B3 disagreement count and hash of disagreement IDs.

It must not require B3 equality. This amendment changes no geometry, curve, N grid, threshold, Stage-1 authorization, or control.