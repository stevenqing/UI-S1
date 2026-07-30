# Preregistration Amendment 002: Out-of-Domain Parsed Points

Date: 2026-07-30

Status: applied after W1 smoke failure and before any W1 benchmark result.

## Discovered constraint

The released Mind2Web parsers mark 1,151 visual predictions as parsed while returning coordinates outside `[0,1]^2`: ShowUI-2B has 1, Qwen2.5-VL-3B has 890, and Qwen2.5-VL-7B has 260. Reclassifying these rows as parse failures or clipping coordinates would violate the inherited parser contract and amount to output repair.

## Amended inference-kernel domain

The preregistered triangular formula is extended to all finite points without changing its fixed scale:

`kappa_coord(p, q) = max(0, 1 - ||p - q||_2 / sqrt(2))`.

- `sqrt(2)` remains the diameter of the valid normalized output domain and is not fitted.
- Out-of-domain coordinates are preserved exactly.
- No clipping, renormalization, or parse-status change is allowed.
- A point has self-similarity one, but distant invalid points have zero pair similarity.
- The unchanged released evaluator determines final success, so out-of-domain aggregate outputs remain failures unless the aggregate itself returns to the valid target region.

This amendment is necessary for the full-scope ablation. The preregistered deployable Mind2Web scope excludes both Qwen base lanes but still includes the single ShowUI-style possibility only if its model passes the capability band.