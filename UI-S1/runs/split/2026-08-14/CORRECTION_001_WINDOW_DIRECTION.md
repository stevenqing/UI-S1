# SPLIT Correction 001: Target-window direction

Date: 2026-08-14
Scope: geometry implementation only; no model forward was run.

The first geometry audit attempt placed a target center on the image-side edge that was farther from the other mode, causing the crop itself to extend toward the other mode. This contradicted the frozen requirement that the target window include its own mode while excluding the other mode.

The corrected construction makes the window extend away from the other mode:

- target left/above other: target is one pixel inside the right/bottom edge;
- target right/below other: target is one pixel inside the left/top edge.

The invalid artifact is retained as `FAILED_GEOMETRY_ATTEMPT_001.json`. It reported 997/1,187 failures and must not be used for Z-K6 adjudication.
