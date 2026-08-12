# Amendment 013: Formal TriVUS Runner Primitives

Date: 2026-08-12

Timing: after metric-free real-data smoke commit `e6f63423013a39c0d7661fa975756970a534c7d8`, before any TriVUS optimizer step, checkpoint, threshold, outer-label access, or model result.

## 1. Seven fit specifications

Each inner/final split fits exactly seven models:

- `JOINT3`;
- `TARGET_ONLY_MIND2WEB`;
- `TARGET_ONLY_SCREENSPOT_PRO`;
- `TARGET_ONLY_ANDROIDCONTROL`;
- `JOINT2_NO_ANDROID`;
- `NO_VISUAL`;
- `RANDOM_ID_PLACEBO`.

The three target-only predictions are composed by family into one `TARGET_ONLY` control. Every model in the same split independently resets the identical split seed; no model-specific seed offset or warm start is allowed.

## 2. Epoch mechanics

An epoch includes every positive-weight active model-training row exactly once in a deterministic sample-key-derived SHA-256 order. Candidate slots receive a deterministic sample-key-derived 12-slot permutation each epoch. K=3 rows permute all 12 positions, preserving the mask and zero padding through the frozen model permutation function.

Each mini-batch loss uses the full epoch's positive weight sum as normalization. Gradients accumulate over all mini-batches, are clipped once, and AdamW steps exactly once per epoch.

Training and checkpoint data must have disjoint sample/context keys and disjoint fold sets before standardization or optimization.

## 3. Checkpointing

Checkpoint loss is the summed normalized S2 loss over positive-weight active checkpoint rows. At most 30 epochs run. Improvement must exceed `1e-5`; patience is five. Exact/non-improving ties retain the earliest epoch. The selected state is copied to CPU. Final epoch is the half-up median of four selected inner epochs, separately for all seven fit specifications.

## 4. Predictions and controls

Prediction records contain context/sample/family/cell/row/fold/group metadata, direct and fallback indices, changed bit, candidate-minus-KEEP margin, fallback-wrong sigmoid score, and direct/fallback success bits. Padded candidates cannot be selected. A TARGET_ONLY model rejects data outside its single family. `TARGET_ONLY` composition requires exact expected context-key coverage per family and no duplicate context keys.

## 5. Safe thresholds

Each model policy independently selects thresholds from infinity, zero, and deciles of strictly positive changed-row margins/wrong scores. Cells are four arms each for Mind2Web and ScreenSpot-Pro plus Android Low/High.

Family pooled candidates require every cell delta at least `-0.5*MDE` and equal-cell mean at least `-0.25*MDE`. A cell with at least 200 changed opportunities selects its own point-maximizing threshold subject to `-0.5*MDE`; otherwise it uses the family pooled threshold. Ties choose larger fallback-wrong threshold, then larger margin threshold.

## 6. Execution boundary

This amendment authorizes only pure/synthetic primitives and tests. It does not authorize real optimizer steps. The physical outer runner, pretest seal, outer-label gate, launcher, and finalizer must be committed and tested result-free before formal execution authorization.