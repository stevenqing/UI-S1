# Amendment 004: H1 Kernel Scale and M2 Risk Rule

Date: 2026-07-31

Status: frozen before H1 candidate generation.

ScreenSpot-Pro point-in-bbox has target-dependent box geometry and supplies no fixed inference-time radius. H1 therefore uses the official MVP grouping threshold, 14 original-image pixels, as the fixed coordinate-kernel scale:

`u_ij = exp(-pixel_distance(c_i,c_j)^2 / (2 * 14^2))`.

No target bbox enters the kernel. Because CCM bins pair similarity by development-fold empirical ranks, every strictly monotone transformation of this `u` preserves candidate bins and decisions, up to tied values.

M2 uses nested grouped folds separately for each N. For outer test fold `f`, threshold-dev is `(f+1) mod 5`; the remaining three folds fit the eight-bin add-one LR table. Candidate thresholds are all finite nonnegative threshold-dev gaps between M1 winner score and full-image candidate score plus infinity. Select the smallest threshold whose threshold-dev accuracy is at least the full-image accuracy. At test time use M1 only when `gap >= threshold`, otherwise return the full-image candidate. No test label enters calibration or threshold selection.