# Amendment 002 — Exact Cross-Fitted CEV Reference

Date: 2026-08-11

Status: `PRE_RESULT`

The Utility-LSA specification requires utility labels relative to the exact frozen CEV-A fallback. Before any Utility-LSA result was produced, the executable meaning is clarified:

1. Every row uses the CEV policy from that row's own upstream outer fold.
2. Its CEV source reliability is fitted on the other four folds, exactly as in the frozen upstream evaluation.
3. Its CEV granularity, scale, threshold, and arm configuration come from that same upstream fold.
4. This fallback is invariant to Utility-LSA's current outer or inner split.
5. Utility-LSA candidate features still use only the current model-training folds; frozen fallback reliability is never exposed as a learned feature.

This creates a pre-existing cross-fitted reference policy for every row and avoids two errors: fitting the reference on the row itself, or changing the reference across Utility-LSA OOF iterations. All row-level reconstructed fallback correctness values must match frozen CEV outputs before training proceeds.
