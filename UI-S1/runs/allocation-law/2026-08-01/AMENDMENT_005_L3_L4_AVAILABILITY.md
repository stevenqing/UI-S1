# Amendment 005: L3/L4 Availability Operations

Date: 2026-08-01

Status: frozen after L1/L2 results and before producing L3/L4 status artifacts. No L3 pooled result exists. L4 E2 has not been scored.

## L3

A view counts toward the L3 V-only pool only when its crop region is produced by an attention proposal pipeline. Existing Collision-Law Mind2Web W2 views are audited but not substituted: v1 is padding, v2/v3 are full-prediction-centered fixed-fraction crops, and v4 is a lower-token full image. They are not attention proposals.

The Mixed pool requires three distinct deployable lineages with four aligned views each. Full-view-only predictions do not satisfy this requirement. Lineage-only requires 12 eligible aligned full-view candidates exactly as stated in the spec; fewer candidates are not padded or duplicated.

If any required fixed-budget pool is incomplete, L3 is reported as `BLOCKED_PREREGISTRATION_GAP`; L-K4 remains `NOT_EVALUATED`, not passed or triggered. No model, crop heuristic, attention layer, or query token is selected after observing a pooled L3 outcome.

## L4

The primary proposer diagnostic is the fraction of candidate regions that fully contain the GT bbox. GT-center containment and per-rank rates are secondary diagnostics. These fields are used only after proposal generation and never by inference.

E1 uses the frozen GTA1 official attention-ranked N12 pool with layer 20 and query token comma. Its B3, M1, and pass@12 outcomes are reused from L1 V-only N12 because candidate identities, order, folds, and aggregator implementation are identical.

E2 requires a native released proposer for every selected model. Missing extraction support or architecture-incompatible conversion for either Qwen3-VL-8B-Instruct or UI-TARS-7B-SFT makes E2 unavailable. No generic attention hook, GTA1 region reuse, random crop, or other heuristic replacement is permitted.
