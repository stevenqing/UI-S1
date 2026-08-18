# OWIN oracle candidate coverage semantics

Date: 2026-08-17

Status: `FROZEN_DURING_GENERATION_BEFORE_ANY_COMMON_OUTPUT_PARSE_OR_ENDPOINT`

OWIN oracle windows are GT-constructed geometry and have no GTA1 attention-coverage score. Canonical B3 requires a public `coverage` value only for group and representative tie-breaking.

All 12 oracle-pool candidates receive `coverage=0.0`, including the full image and all 11 oracle crops. This follows the existing Allocation-Law convention for generated candidates without a retained proposer-coverage channel. No coverage score may be copied from a different window, inferred from GT, fit from correctness, or derived from model output.

Candidate order is fixed slot order 0 through 11. Therefore B3 ties use canonical slot order after group size. M1_ccm uses the same candidates with model source `GTA1-7B` and their fixed slot/view indices.

This note was written while common generation was still running and before any OWIN trace was parsed for coordinates or correctness. It resolves an implementation field absent from the prose specification without selecting among observed outcomes.