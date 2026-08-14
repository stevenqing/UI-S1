# SPLIT Amendment 002: Window integerization

Date: 2026-08-14
Timing: after `ZERO_GPU_GATE.json` commit `84c6dcb`, before constructing any SPLIT window or observing any geometry failure statistic.

## Trigger

The preregistered side length $L$ can be non-integer, while PIL crop boxes and the one-pixel edge inset require a frozen integer convention.

## Frozen resolution

- Compute continuous $L$ exactly as preregistered, then use `side = ceil(L)` capped at `min(width, height)`.
- Window coordinates are integer `[left, top, right, bottom]` boxes with `right-left = bottom-top = side`.
- Point containment is left/top inclusive and right/bottom exclusive.
- A target center is represented by its original floating-point candidate mean. Integer window origins are chosen with floor/ceil as needed to keep the center at least one pixel inside the edge away from the other mode, then clamped while preserving `side`.
- After clamping, the explicit include/exclude predicates decide validity. No rescue or alternate axis is attempted.
- Processor resize metadata uses `smart_resize(side, side)` with the model processor's `patch_size * merge_size`, `min_pixels=1,000,000`, and `max_pixels=4,000,000`.

All other geometry rules remain unchanged.
