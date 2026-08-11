# Amendment 002: Fallback-Agnostic Nested Anchor

Date: 2026-08-11

Timing: frozen before any anchor labels were opened or anchor result was computed. The first blind run was stopped after 10,278/14,644 records when this issue was identified. Its eight partial shards are retained under `INVALID_SECOND_LEVEL_LEAKAGE/` and are prohibited from adjudication.

## Problem

The stopped run placed each row's row-cross-fitted CEV-A fallback in the prompt. When that row serves as inner development data for a different outer fold, its row-cross-fitted fallback may have been fitted using the outer-test fold. Because the fallback changed the prompt, correcting only the adjudicator would not remove second-level stacking leakage.

## Correction

1. The blind visual anchor is fallback-agnostic. Its prompt contains the screenshot, task/history, and 12 permuted candidates, but no CEV identity and no KEEP label.
2. The anchor outputs one-step logits only for A--L. The direct candidate is the largest A--L probability.
3. Exact nested CEV fallback indices are generated in a CPU-only sidecar for every `(outer_fold, sample_key)` context:
   - outer-test rows use the frozen outer-fold CEV-A policy fitted on the other four folds;
   - each inner-holdout development row uses `fit_inner_policies` on the other three outer-development folds, exactly matching Utility-LSA's second-level nesting.
4. The safe-policy scores become fallback-relative after inference:
   - candidate margin: `p(direct) - p(fallback)`;
   - fallback-wrong score: `1 - p(fallback)`.
5. Threshold selection remains exactly as frozen in Amendment 001, but it reads only the nested contexts for the current outer fold.
6. The 10,278 stopped predictions cannot be reused because their prompts named the fallback. All 14,644 fallback-agnostic predictions are recomputed from clean output paths.

This amendment changes only the eligibility anchor. It does not expose labels to inference and does not alter VUS V1--V5.
