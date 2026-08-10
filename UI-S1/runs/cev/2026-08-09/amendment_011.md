# Amendment 011 — Post-Leakage Reconstruction of CEV / CEV-A

Date: 2026-08-10

Status: `FROZEN_BEFORE_CEV_RESULTS`

## 1. Why this amendment exists

The 2026-08-09 handoff states that a CEV/CEV-A specification had been frozen, but no canonical CEV spec, Amendment 011, preregistration YAML, or `runs/cev/2026-08-09/` directory exists in the repository or indexed Copilot session history. The original document is therefore treated as lost.

This amendment reconstructs the experiment rather than pretending to recover the original verbatim. The reconstruction is auditable and deliberately conservative.

## 2. Information already observed

Before this amendment, the following ScreenSpot-Pro C-uni cells had been observed:

| Rule | Accuracy |
| --- | ---: |
| A2 density medoid | 63.8836% |
| Complete-link + candidate votes | 63.8836% |
| Complete-link + lineage dedup | 63.0614% |
| Single-link + lineage dedup | 62.5553% |
| Single-link + candidate votes | 63.2511% |

These values are contamination anchors. They cannot be used for threshold selection, method optimization, or confirmatory claims. V1 uses row-wise A2 reproduction solely as an implementation check.

Known F1 results also motivate the experiment and are not new evidence: Mind2Web favors majority, while ScreenSpot-Pro favors coordinate density.

## 3. Reconstruction choices

The lost details are replaced by the following frozen choices:

1. CEV means deterministic complete-link candidate voting over a fixed equivalence ladder.
2. Main voting is candidate-level, because the leaked EQV self-check already established that unconditional lineage dedup damages 7B ScreenSpot-Pro.
3. CEV-A may select granularity globally or by predicted action, but only through nested development folds.
4. The granularity ladder is fixed to G0–G4; no learned continuous mixture is introduced.
5. ScreenSpot-Pro uses the inherited 14 px scale. Mind2Web geometry and parameter thresholds are selected from fixed grids using inner-development data only.
6. The mandatory benchmark-specific dev-selection control uses the same nested split.
7. All outputs are real candidates; no continuous synthetic point is generated.

These are new reconstruction decisions, not recovered original text.

## 4. Integrity rules

- Canonical contract: `configs/cev_prereg.yaml`.
- Human-readable contract: repository root `cev-spec-2026-08-09.md`.
- This amendment and both contracts must be committed before any CEV result file is created.
- Existing candidate banks may be read only after the preregistration commit when computing CEV outcomes.
- No GPU/model inference is permitted.
- PID 2274 remains protected.
- Every output must be backed up outside git with SHA-256 retention.

## 5. Interpretation

The strongest possible conclusion is limited to a method contribution under V4. If CEV-A only matches nested dev-selection, its contribution is explanatory. If it loses, F1 remains primary and CEV is reported as a failed unification attempt.

Regardless of outcome, the paper must disclose that this is a post-leakage reconstruction and that P-A/P-B are contaminated diagnostics.
