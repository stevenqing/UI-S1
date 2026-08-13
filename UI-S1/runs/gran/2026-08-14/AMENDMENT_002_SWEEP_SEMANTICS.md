# Amendment 002: Sweep Semantics

Date: 2026-08-14

Timing: after input, CLICK-scope, and implementation-anchor locks; before any $\hat p$, margin, contamination, or $\tau$ sweep result.

The original specification leaves several implementation choices underdetermined. They are frozen as follows.

1. ScreenSpot-Pro uses the complete 36-action bank: three lineages by views 0--11. The 12-candidate E1 C-uni value remains an implementation anchor only.
2. Mind2Web evaluates all four 12-candidate arms. G-P2's primary arm is C-uni; C-cond, C-rand, and C-self are secondary robustness analyses.
3. Candidate coordinates are divided by the image diagonal before clustering. Mind2Web normalized positions are first mapped to image pixels, then divided by the same diagonal.
4. Mind2Web clustering is action-type first and spatial complete-link second. G-P2 includes rows whose ground-truth action is CLICK; candidates of a different predicted action cannot join a CLICK block.
5. The prior endpoint is the highest inner-training correctness source, with lexical source identity as the final tie-break. The same rule supplies $\pi$ inside a winning density block.
6. $\tau$ is selected by inner-validation micro accuracy. Exact ties use this fixed order: exact-coincidence endpoint, finite grid from smallest to largest, then single-block endpoint. A finite boundary winner still triggers G-K6.
7. The density margin is density-policy correctness minus prior-policy correctness on the same held-out row. All layer curves and zero-point analyses use this paired binary difference.

No observed label statistic informed these choices. The only executed labels so far are the predeclared E1 anchors and the action-type count used to lock the number of strata.