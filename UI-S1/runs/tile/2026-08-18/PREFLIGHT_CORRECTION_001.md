# TILE preflight correction 001

Date: 2026-08-18

Status: `DECLARED_BEFORE_ANY_TILE_STAGE0_STATISTIC`

The first successful preflight reproduced row/fold/layout anchors but omitted Amendment 002 and `configs/amendment_002.yaml` from its dependency hash table. No TILE eccentricity pair, curve, row score, ledger, or gate had been computed.

The incomplete preflight is retained under `failed_attempts/preflight_missing_amendment_002/PREFLIGHT.json`. The corrected preflight binds both baseline-reconciliation artifacts in addition to all prior dependencies. This changes no data, baseline, geometry, curve, endpoint, or threshold.