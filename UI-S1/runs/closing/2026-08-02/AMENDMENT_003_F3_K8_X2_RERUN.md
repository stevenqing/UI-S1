# Amendment 003: F3-Triggered K8 X2 Rerun

Date: 2026-08-02

Status: frozen after F3 outcome `anchor_pass_microchain_length_sensitive` and before any K8 X2 inference or result. This amendment follows the predeclared `anchor_pass_microchain_length_sensitive` action in `configs/f3_outcomes.yaml`.

## Estimand

The original N12 fixed-view Q1-Q4 design cannot be extended to K8 at the same budget because GTA1 has only 16-19 unique fixed views. The K8 rerun therefore estimates a cleaner factorial intervention within stochastic UI-Zoomer candidates: lineage allocation (single versus mixed) crossed with adaptive zoom (disabled versus enabled), holding K8 global samples and attempted candidate budget fixed.

## Nine-forward paired chain

For each model/identity/chain:

1. Generate eight full-image samples jointly with temperature 0.9 and top-p 1.0 using the frozen UI-Zoomer prompt and token-confidence extraction.
2. Compute the official K8 gate, 50-pixel point-to-box adapter, top-75% variance crop, sigma 2.5, and minimum crop side 512.
3. Always generate an independent ninth full-image sample at temperature 0.9 (`global_confirmation`).
4. If the gate is unreliable and a crop exists, also generate one deterministic crop refinement. This extra generation is a counterfactual branch artifact; no evaluated cell uses both the confirmation and refinement.
5. The fixed candidate chain is the eight global samples plus global confirmation. The adaptive chain is the same eight global samples plus crop refinement when triggered, otherwise the same global confirmation.

Thus every evaluated chain has exactly nine attempted forwards and no candidate duplication. Raw union generation may use ten forwards for an uncertain chain to make fixed and adaptive branches paired; this does not change either cell's 9-forward accounting.

## Four cells

Each cell has 27 candidates:

- Q1-K8 single/fixed: three independent GTA1 chains, fixed branch.
- Q2-K8 single/adaptive: the same three GTA1 chains, adaptive branch.
- Q3-K8 mixed/fixed: one chain each from GTA1, Qwen3, and UI-TARS, fixed branch.
- Q4-K8 mixed/adaptive: the same three lineage chains, adaptive branch.

Q1/Q2 candidate order is chain-major. Q3/Q4 order is slot-major, then model in frozen order GTA1, Qwen3, UI-TARS. For mixed pools, `view_index` is slot 0-8 so cross-lineage same-slot pair semantics remain defined.

## Analysis

Use the same B3 and fold-local M1 implementations, application folds, failure kappa, and 10,000 paired fold-stratified bootstrap interaction as X2, now at budget 27. Primary success still requires Q4-K8 highest and interaction nonnegative. X-K1 remains interaction 99% CI upper bound below zero.

The K8 result replaces the K3 X2 conclusion in Closing and paper writing. The original K3 result remains an archived budget-normalized observation and does not enter R1-R4.

## Isolation and failures

Trace fields contain no target bbox. Seeds are SHA-256 derived from row, cell family, model, chain, stage, and base seed 20260802. Parse failures remain raw nulls and use the frozen `(0,0)` evaluation sentinel without replacement inference. All model revisions, hashes, seeds, boxes, gates, crops, branch selection, and 27-candidate cell budgets fail closed before scoring.
