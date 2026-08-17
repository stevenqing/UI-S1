# Research Process Disclosures

Updated: 2026-08-17

This list records specification, evidence-status, and execution-boundary errors that materially affect interpretation. Original results and correction artifacts remain immutable.

## GRAN

1. The original four-stratum ScreenSpot-Pro requirement was arithmetically infeasible at a 400-row minimum because $1,581<4\times400$. It was corrected to 395 before the affected statistics. See `runs/gran/2026-08-14/AMENDMENT_001_STRATUM_FEASIBILITY.md`.
2. Bank scope, coordinate normalization, action-first clustering, tau tie order, and density-margin semantics were underdetermined in the original specification and frozen before the sweep. See `runs/gran/2026-08-14/AMENDMENT_002_SWEEP_SEMANTICS.md`.
3. GRAN Assumption A2 treated agreement kappa as an ICC-like count correlation without direct validation. G-P8 remained `NOT_ADJUDICABLE_PREREG_UNDERDEFINED`; no historical GRAN status is changed by later diagnostics.

## MASK

The draft research sequence risked conflating UI-TARS as a frozen bank lineage with the model intended for masked forward calls. The committed MASK specification resolved the roles before execution: GTA1 was the sole proposed forward model, UI-TARS remained a bank lineage, and MASK ultimately ran zero model forwards. This is a corrected model-role ambiguity, not an executed wrong-model result. The Stage-1 kappa-cache correction was computational only and preceded `STAGE1.json`; see `runs/mask/2026-08-14/CORRECTION_001_STAGE1_KAPPA_CACHE.md`.

## OTEXT

The proposed confirmatory framing was invalid because ORTH had already used all 1,581 ScreenSpot-Pro labels to select the OCR/text direction. OTEXT was downgraded before its statistics to `POST_SELECTION_VALIDATION`; nested evaluation cannot restore confirmation on the same rows. See `runs/otext/2026-08-14/SPEC.md`.

## XSCR

The nominal prospective holdout was excluded from reported aggregates, but all private-label files were parsed during input locking and loaded during Q3/Q4. It was therefore not unread and cannot support independent validation. See `runs/xscr/2026-08-14/CORRECTION_004_HOLDOUT_LABEL_ACCESS.md`.

## EVID

The fixed EVID constants 0.895 and 0.398 were described as prior kappa anchors suitable for the effective-evidence formula. They are AndroidControl failure-agreement kappas, while the formula requires a ScreenSpot-Pro error-correlation model. The benchmark and statistical estimand differ, and GRAN had already marked this kappa-as-ICC mapping as requiring validation. Therefore `EVID_FIXED_AGGREGATOR_FAILED_STAGE2_BLOCKED` strictly rejects the frozen aggregator with these transferred constants, not every correctly parameterized member of the score family. EVID's result, gates, and Stage-2 block remain unchanged. ICC is a separate retrospective diagnostic and remains post-selection.

## CWIN

The initial `STAGE0.json` schema retained all-K geometry but reported L1/L3 summaries only for nested-selected K, while the specification required those endpoints for every K in `{2,3,4}`. After Stage 0, `STAGE0_REPORTING_RECOVERY.md` and its reconstruction code were committed before the omitted tables were computed. `STAGE0_ALL_K.json` reconstructs only all-K L1/L3 from the frozen geometry and candidate bank; K=4 reproduces the original selected-K result. It changes no geometry, nested selection, L2/L4 gate, W-G1, W-K5, or authorization.

The first scratch-retention implementation copied artifacts one file at a time and was interrupted after persistent backend request latency. Its 304 partial copies were retained rather than deleted. `RETENTION_RECOVERY.md` was committed before a single-tar recovery, which records every source artifact's SHA-256 and independently verifies every archive member. This retention recovery changes no scientific result. GPU remains unauthorized.

## OWIN

OWIN's base draft described five leaked ScreenSpot-Pro cells as if row IDs could be removed from sampling. They are disclosed fold-level aggregate values, not row identities; the committed base specification corrected this before any OWIN output. No fictitious row exclusion is performed.

Amendment 002 corrects two additional result-free drafting errors. First, the historical `85.77%` quantity is the fraction of rows whose target center has positive existing crop coverage, not a model-success rate; frozen full B3 success is 63.69%. Any future net-benefit ledger must report damage separately on original-correct and crop-covered rows. Second, radius calibration instantiates candidate windows around GT bbox centers and is therefore evaluation-side GT geometry calibration, not label-free computation, even though it reads no correctness or model output. These corrections preceded every OWIN preflight, geometry output, model forward, and GPU authorization.