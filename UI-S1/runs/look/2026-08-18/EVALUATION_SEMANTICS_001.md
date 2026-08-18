# LOOK evaluation semantics 001

Date: 2026-08-18

Status: `FROZEN_BEFORE_ANY_LOOK_GPU_OUTPUT`

L-P1/L-P4 and sensitivity AUROC use only contexts with a parsed finite full-image coordinate, matching the specification's “valid records” wording. An unmappable context contributes no candidate records to AUROC, while its row remains in L-P2/L-P3 and counts selected-mode correctness as false. L-K4 separately gates the unmappable fraction.

Distance normalization uses the original image diagonal read from the committed image file, never crop dimensions or window bounds. This note resolves implementation details before GPU authorization and changes no sample, window, endpoint, or threshold.