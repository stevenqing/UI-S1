# Preregistration Amendment 001: Mind2Web Coordinate Kernel

Date: 2026-07-30

Status: applied before W1/W2 result generation.

## Discovered constraint

All eleven released Mind2Web traces contain the ground-truth bbox as the row-level `bbox` field. Their released parsers output only a predicted point (or no point); none outputs a predicted element bbox. Consequently, the proposed pairwise kernel `1[point in bbox]` cannot be evaluated between two predictions without reading the test ground truth.

Using the row's GT bbox inside PKA would leak the answer and is prohibited. This is a representational limitation of point-only agent traces, not a missing implementation detail.

## Amended contract

Mind2Web now has two explicitly separated coordinate kernels:

1. **Analysis kernel (GT-only, never available to an aggregator):** evaluator indicator `1[predicted point in GT bbox]`. This is used only for post-hoc truth concentration, collision measurement, stratification, and evaluator scoring.
2. **Inference kernel (GT-free PKA main path):** triangular similarity on the normalized unit-square coordinate domain,

   `kappa_coord(p, q) = max(0, 1 - ||p - q||_2 / sqrt(2))`.

The denominator `sqrt(2)` is the fixed diameter of `[0,1]^2`; it is not fit on dev or test data. The kernel has no bandwidth or temperature. Missing coordinates contribute zero coordinate similarity except for parameter-free action types, where the coordinate factor is omitted and the preregistered `rho_0` normalization applies.

## Claim boundary

- The exact evaluator-kernel claim remains valid for AndroidControl because its metric supplies a fixed radius and coordinate distance.
- On Mind2Web, PKA is **domain-kernel aligned** rather than exactly evaluator-kernel aligned. The paper must state this limitation in the main method section.
- K2 remains unchanged and is evaluated on the actual Mind2Web Step SR plus W2 MDE.
- No result observed after this amendment may be used to choose another Mind2Web inference kernel. Smooth signed-distance kernels remain GT-only analyses and cannot be aggregator inputs.

## Verification requirement

Unit tests must fail if a Mind2Web inference-kernel call accepts or reads a GT bbox. Analysis-kernel functions must be named with a `_gt_analysis` suffix and cannot be imported by `pka.py`.