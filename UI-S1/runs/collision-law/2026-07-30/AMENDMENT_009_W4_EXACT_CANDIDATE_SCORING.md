# Amendment 009: W4 Exact-Candidate Scoring Correction

Date: 2026-07-31

Status: post-result implementation correction. No inference was repeated and no method, threshold, calibration table, source set, or candidate selection changed.

## Defect

Released W4 model responses encode point actions as two-value `bbox_2d: [x, y]`. The initial W4 A3 and CCM confirmation analyzers parsed an exact source candidate and serialized it as a four-value degenerate box `[x, y, x, y]` before calling the official scorer. The official scorer does not treat these representations identically. This changed 13 UI-AGILE-3B Low baseline rows and 251 UI-AGILE-7B High baseline rows from failure to success.

## Correction

A3 and CCM select an existing candidate exactly. They now submit that source's original released response to the official scorer. A1, A2, and A4 synthesize new outputs and retain the declared serialization path. Reporting-only W4 A0 comparison is computed from each held-out fold's original source response and cannot affect CCM calibration or selection.

The invalid intermediate W4 JSON was never committed. Corrected confirmation baseline Step SR exactly equals the independently scored source artifact in both settings.