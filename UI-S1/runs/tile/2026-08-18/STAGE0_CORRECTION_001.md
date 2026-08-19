# TILE Stage-0 correction 001

Date: 2026-08-18

Status: `DECLARED_AFTER_FIRST_STAGE0_BEFORE_FINAL_STAGE0_COMMIT`

The first Stage-0 run correctly used V-only B3 for N selection, primary ledgers, T-G1, and T-G2. It also wrote correct row-level C-uni expected repair/damage/net fields.

The C-uni `original_correct` contextual summary mistakenly called the generic V-only summary helper, so that one contextual subgroup table used V-only row fields. Primary values, all gates, selected N, raw pairs/curves/scores, and the C-uni full-benchmark contextual sum were unaffected.

All first-run artifacts are retained under `failed_attempts/stage0_contextual_summary_001/`. The correction lets the summary helper take an explicit field prefix and recomputes C-uni contextual domains from `C_uni_expected_*`. No rule, curve, fold, score, grid, threshold, or authorization changes.