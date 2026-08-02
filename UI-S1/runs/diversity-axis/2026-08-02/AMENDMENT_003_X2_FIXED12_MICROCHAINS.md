# Amendment 003: X2 Fixed-12 UI-Zoomer Microchains

Date: 2026-08-02

Status: frozen before any X2 model inference or X2 result. Supersedes only the blocked budget-operation field in the result-free configs; X2 predictions and kill conditions are unchanged.

## Motivation and claim boundary

Official UI-Zoomer uses eight stochastic global samples and one conditional deterministic crop refinement, so it consumes eight or nine forwards. It cannot populate a fixed-12 candidate pool without either wasting budget, duplicating candidates, or defining a budget-normalized extension.

X2 therefore uses an algorithm-level `fixed12_microchain` extension. It preserves UI-Zoomer's uncertainty gate, variance crop, and rule that zoom occurs only for uncertain cases, but uses three rather than eight global samples per gate. The official K=8 ScreenSpot-Pro anchor remains a separate unavailable sanity check; X2 is not labeled an exact UI-Zoomer reproduction.

## Four-forward microchain

Each microchain consumes exactly four useful model forwards:

1. Generate three full-image samples independently with temperature 0.9, top-p 1.0, and deterministic per-row/per-chain/per-slot seeds.
2. Convert each valid point to a 50x50 pixel box, clipped to the original image. Confidence is the geometric mean probability of generated tokens, computed from normalized transition scores.
3. Compute official UI-Zoomer spatial consistency (ordered mean pairwise box IoU) plus mean token confidence. The gate is reliable only when the sum is strictly greater than 1.5.
4. If unreliable and at least one valid candidate exists, use the official top-75%-by-center-distance variance-decomposed square crop with sigma 2.5 and minimum side 512 pixels, then run one deterministic crop refinement. If reliable, or if no valid crop can be formed, spend the fourth forward on an independent full-image sample at temperature 0.9. No forward or candidate is duplicated.

All four outputs remain candidates for B3/M1/pass@12. Parse failures consume their forward and are retained as invalid records but excluded from the 12-candidate evaluation pool; X2 fails closed unless every identity has 12 valid candidates.

## Cell construction

- Q2: three independent GTA1-7B microchains, ordered chain-major. Candidate `view_index` is `4 * chain + slot`.
- Q4: one microchain each for GTA1-7B, Qwen3-VL-8B-Instruct, and UI-TARS-7B-SFT. The merged order is slot-major then model in the frozen order GTA1, Qwen3, UI-TARS; each model's `view_index` is the slot 0-3.

Q1 and Q3 remain the existing fixed-view N12 pools. New candidates use coverage zero for B3, matching the prior generated-model contract.

## Backend and isolation

GTA1 uses its resolution-aware MVP prompt and pixel-coordinate parser. Qwen3/UI-TARS use the frozen H3 normalized-point prompt and parser. The point-to-box adapter affects only gating/crop geometry; model output remains a point. New traces contain no target bbox. Labels are joined only after identity, revision, point, budget, seed, and candidate-hash validation.

Seed base is `20260802`; concrete seeds are the first eight bytes of SHA-256 over `row_id|cell|model|chain|slot|seed_base`, interpreted as an unsigned integer and reduced to the backend-supported range.