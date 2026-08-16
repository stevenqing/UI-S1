# Correction 001: report KaTeX escape

Date: 2026-08-15

Status: `FORMATTING_ONLY_NO_RESULT_CHANGE`

The generated report rendered the transition label `2\to3` with a tab because the Python template used an unescaped `\t`. This correction replaces only that display string with valid KaTeX. No number, gate, decision, result, or claim changes.

The original retention manifest remains immutable. A supplemental correction manifest retains the corrected report, adjudication, status, and this correction note.