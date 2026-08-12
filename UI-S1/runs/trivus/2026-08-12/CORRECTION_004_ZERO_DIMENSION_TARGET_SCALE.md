# Correction 004: Zero-Dimension Target Scale

Date: 2026-08-12

Timing: after Git-root correction commit `d10a93bc17902342e114227551a561bc2d81e248` and corrected private-scale authorization commit `9cef05f01dce353520aab54082518294ff9d7a98`, before any private-scale output directory, fallback context, performance metric, or model fit.

The corrected private-scale invocation consumed its one-time authorization and then failed while deriving the scale for Mind2Web row `6b215dbb-a2c4-451c-9c34-9bafe6660c14__e285a857-cb15-4f89-9693-ac66aaa53313`. Its frozen target box has width 231 and height 0 on a 1280 by 720 image, yielding normalized scale `(0.18046875, 0.0)`.

The new implementation had incorrectly required each normalized target dimension to be strictly positive. The frozen CEV implementation applies no such per-row restriction: it retains each normalized width/height and computes the fit-fold median. Exact reproduction therefore requires finite nonnegative dimensions, including zero.

The failure occurred after authorized private target access and authorization-receipt creation, but before staging or publishing any private-scale fold. No fallback context or performance metric was computed. PID 2274 was not altered.

The correction changes both scale sealing and scale loading from finite-positive to finite-nonnegative validation and adds a regression test for the exact observed zero-height value. Negative or non-finite dimensions remain rejected. Authorization `9cef05f` is consumed and cannot be replayed; a new implementation commit and fresh authorization nonce are required.