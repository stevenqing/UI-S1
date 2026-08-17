# CWIN Stage-0 reporting recovery

Date: 2026-08-17

Status: `DECLARED_AFTER_STAGE0_BEFORE_ALL_K_SUPPLEMENT`

Stage 0 completed before this declaration. Its frozen primary outputs are:

- selected K: 4 on all five outer folds;
- nested outer-test L4-upper: `0.20619860847564833`;
- W-G1: pass;
- W-K5: triggered;
- GPU authorized: false.

An output-schema omission was found during result audit. `SPEC.md` requires L1 coverage transitions and L3 drop-only B3/M1 changes for every K in `{2,3,4}`. `STAGE0.json` reports those endpoints only after nested K selection. The all-K geometry, drop order, and complementary-window sequence were retained in `raw/geometry_all_k.jsonl`, but all-K label summaries were not written.

The recovery is reporting-only:

1. load the committed GTA1 V-only N12 candidates and the retained all-K geometry;
2. reconstruct the frozen drop-only pool separately for K=2, 3, and 4;
3. evaluate canonical B3 and fold-local M1_ccm using the same historical modules and folds;
4. reconstruct target-center coverage transitions from the retained windows and regions;
5. write `STAGE0_ALL_K.json` with per-K L1 and L3 only.

The recovery may not change geometry, drop order, complementary windows, folds, K selection, L2, L4, W-G1, W-K5, controls, or aggregators. It may not authorize GPU or Stage 1. `STAGE0.json` remains immutable and authoritative for the nested gate.