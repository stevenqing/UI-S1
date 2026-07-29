# Unreleased ShowUI Mind2Web Rows Preflight

Status: `TWO_ROWS_BLOCKED_UNPUBLISHED_CHECKPOINTS`

## ShowUI-2B Downstream Fine-Tuned

Paper Cross-Task anchor: `39.9 / 88.6 / 37.2` for Element Accuracy,
Operation F1, and Step Success Rate.

The public `showlab/ShowUI-2B` checkpoint is the generic zero-shot release and
has already been evaluated separately. It is not the downstream Mind2Web
fine-tuned state dict used for this paper row. No official downstream state
dict or adapter is public, so the row cannot be reproduced.

## Qwen2.5-VL-3B-ShowUI

Paper comparison anchor: `43.2 / 88.7 / 39.7` on Cross-Task.

No corresponding official model ID, full checkpoint, or adapter exists in the
showlab or TongUI release channels. The released TongUI-3B model is a different
checkpoint and is already reported under its own identity.

## Resume Conditions

Each row requires an official checkpoint or adapter at a fixed revision and
confirmation of the exact Mind2Web training/prompt contract. Substituting the
public generic ShowUI or TongUI weights would fabricate the model identity.
