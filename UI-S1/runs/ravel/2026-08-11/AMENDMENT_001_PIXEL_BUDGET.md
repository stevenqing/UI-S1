# Amendment 001: Exact Multi-Image Pixel Budget

Date: 2026-08-11

Timing: frozen after pixel-feasibility audit and before any RAVEL model load, logit, or result.

## Feasibility issue

Frozen VUS processed pixels range from 716,800 to 972,800 per row after its 1,600-edge render and original processor limits. On 242 ultra-wide rows, 25% of the VUS budget is lower than the old per-image minimum of 200,704 pixels. Three images therefore cannot satisfy 50/25/25 while retaining that single-image minimum.

## Frozen allocator

1. Compute row budget $B$ exactly from the original image dimensions using the frozen VUS path:
   - scale longest edge to at most 1,600;
   - Qwen factor 32;
   - `min_pixels=200704`;
   - `max_pixels=1003520`.
2. RAVEL processor uses `min_pixels=100352` and `max_pixels=1003520` for every image.
3. Main and random-center modes receive targets `0.50B`, `0.25B`, `0.25B` for global, fine, and context images.
4. Each target is independently rounded down by factor-32 smart resize while preserving aspect ratio. Unused rounding budget is not reallocated.
5. Actual processed pixels are reconstructed from `image_grid_thw` and must be no more than `1.02B`; violation is RAVEL-K1.
6. `global_only`, `fine_only`, and `context_only` each use one image with target `B`.
7. The frozen VUS full-screen overlay logits are the compute-matched full-screen control; no duplicate inference is run for that control.

All RAVEL modes use one Qwen3-VL invocation per row-arm. Pixel counts and ratios are stored in every output record and checked before labels are opened.

## Crop rendering

- Mosaics are four columns by three rows.
- Candidate point is centered in its tile through padding when in-frame.
- Missing or out-of-frame coordinates are explicitly marked.
- Random-center control uses a deterministic center from `(sample_key, candidate_index, scale, seed)` and retains the same labels, crop sizes, image dimensions, prompt, and visual budget.
- Labels/crosshairs are rendered after the tile is resized to final mosaic resolution so they remain legible.
