# ORTH Amendment 003: Scoping report protocol

Date: 2026-08-14
Timing: while EasyOCR raw forward is still running, before any Arm 1 derived matching statistic is computed.

The main report must not select a best OCR engine, matcher, minimum length, or edit threshold. For each engine and matcher family (`exact`, `normalized`, `edit`), it reports the minimum-to-maximum range across the complete frozen grid for:

- overall match rate;
- selected-correct, recoverable, and zero-coverage conditional match rates;
- text and icon all-row channel accuracy;
- matched-only accuracy;
- all-row error kappa.

The three main overlap tables correspond to exact, normalized, and edit matcher families and show ranges. Every individual setting remains available in `ARM1.json`; extrema may be labelled with setting IDs in the appendix but may not be promoted as an optimized result.

The human scoping conclusion may choose only one of three next-design directions: (1) preregister OCR confirmatory study, (2) restore/hash full Mind2Web DOM data before further structured-channel study, or (3) stop orthogonal-channel follow-up. Its rationale must cite robust ranges and structural coverage, not a single best grid point. None of these directions changes an existing project status or creates a paper result.