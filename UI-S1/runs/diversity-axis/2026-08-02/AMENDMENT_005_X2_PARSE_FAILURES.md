# Amendment 005: X2 Parse-Failure Semantics

Date: 2026-08-02

Status: frozen after all X2 inference completed and the integrity-only loader stopped on invalid candidates, before any target bbox was loaded into an X2 pool and before any X2 accuracy, kappa, interaction, area stratum, or X6 held-out outcome was computed.

The completed traces contain 18,972 Q2 forwards and 18,972 Q4 forwards. Q2 and Q4-GTA1 have zero invalid outputs. Q4-Qwen3 has 32 invalid outputs affecting 20 identities, predominantly natural-language refusal/no-visible-element responses; Q4-UI-TARS has six out-of-range normalized-coordinate outputs affecting six identities. Values outside the frozen `[0,1000]` normalized contract remain invalid and are not reinterpreted as pixels. No target label was inspected to make this classification.

The original Amendment 003 incorrectly required all 12 forwards to parse into valid candidates. Forward-budget equality, not parse success, is the causal budget constraint. Parse failure is itself a model outcome and must be scored rather than making the entire experiment unavailable.

For evaluation only, every invalid forward is represented by the same `(0,0)` failure sentinel used by the completed H3 Qwen3/UI-TARS adapters. Its region is the recorded full/crop region and coverage is zero. The raw trace retains `point: null`, response, seed, confidence, branch, and hash; it is never rewritten. No replacement inference, candidate duplication, parser relaxation, coordinate clipping, or row exclusion is allowed.

The sentinel participates in B3, fold-local M1, pass@12, and failure-kappa calculations. This intentionally allows format failures to lower accuracy and agreement metrics. Reports must include per-cell/model invalid-forward and affected-row counts. The experiment remains fixed at exactly 12 attempted model forwards per identity.

X6 uses the same evaluation representation, so its unlabeled distance feature includes the `(0,0)` sentinel. The frozen X6 fit is not refit.
