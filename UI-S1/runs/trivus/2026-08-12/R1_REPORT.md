# TriVUS R1 Complete-Bank Headroom Report

Date: 2026-08-12

Outcome: `PROCEED_TO_BLIND_SELECTOR`

## Result

R1 evaluated the complete frozen three-model pool on the paired 2,000-row AndroidControl Low and High samples. Majority is fit fold-locally using exact action plurality and development-reliability candidate tie-breaking. Candidate oracle succeeds when any real candidate succeeds.

| Setting | Fold-local majority | Candidate oracle | Oracle - majority | 99% paired CI |
| --- | ---: | ---: | ---: | ---: |
| Low | 78.20% | 83.20% | **+5.00 pp** | **`[+3.79,+6.32]`** |
| High | 59.90% | 72.25% | **+12.35 pp** | **`[+10.58,+14.29]`** |

Both preregistered gates require a point effect above 1.0 pp and a positive 99% lower confidence bound. Both pass.

Individual candidate accuracies are:

| Setting | UI-AGILE-7B | GUI-R1-7B | UI-R1-E-3B |
| --- | ---: | ---: | ---: |
| Low | 78.60% | 58.95% | 49.35% |
| High | 63.10% | 46.90% | 23.20% |

## Integrity

Before evaluator import or GT access, R1 reparsed all six actual lane artifacts and revalidated per-shard rows, bytes, SHA-256, exact stable-index coverage, reference order, provenance, and ordered row-identity hashes against `RECOVERY_MANIFEST.json`.

The result was independently recomputed from locked lanes and matched the saved JSON exactly. `R1_HEADROOM.json` SHA-256 is `093fcd8241fc3d1609db510629994527e125a598bbc311e921492aaeaefb20b2`.

## Sampling boundary

This is the frozen paired 2,000-row AndroidControl sample, not all 7,650 clean rows. Relative to historical full-lane scores, sample deltas are:

- Low: UI-AGILE +1.04 pp, GUI-R1 +0.74 pp, UI-R1-E +0.84 pp;
- High: UI-AGILE +2.57 pp, GUI-R1 +1.93 pp, UI-R1-E +0.31 pp.

The paired oracle-minus-majority comparison is internally valid on the complete sample because every policy uses the same rows. It cannot be presented as an unbiased full-AndroidControl estimate, especially for High UI-AGILE.

R1 authorizes only blind fallback-agnostic AndroidControl selector inference and bank locking. It does not authorize private-label model training until the data adapter and exact nested-training runner are separately frozen and committed.