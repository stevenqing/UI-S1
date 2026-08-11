# Amendment 005: Auxiliary Score Direction

Date: 2026-08-11

Timing: frozen after a 512-row mechanical backward smoke and before any formal VUS-SR fit or result. The smoke used S1 for one epoch and was prohibited from selection.

The auxiliary BCE target is `fallback_correct = 1`, but the safe gate requires evidence that fallback is wrong. Therefore:

- S2 and S3 use `fallback_wrong_score = 1 - sigmoid(fallback_correct_logit)`;
- S1 has auxiliary weight zero, so its auxiliary head is untrained and must not enter selection; S1 uses `fallback_wrong_score = 1` for every row, reducing its safe gate to the learned candidate-vs-KEEP margin only.

Using `sigmoid(fallback_correct_logit)` directly would reverse the gate. Allowing S1's random auxiliary head into OOF threshold selection would create a noise feature. Both are prohibited before the first formal fit.
