# Amendment 002: Masked-Channel Isolation

Date: 2026-08-11

Timing: frozen after result-free compile, unit, and one-step optimizer smoke tests; before any formal outer-fold fit, outer-label access, or DELTA result.

The implementation audit found that concatenating all channel embeddings into the gate input allowed a masked channel to influence gate weights assigned to active channels. Zero gate mass alone therefore did not establish channel isolation and would invalidate the same-capacity controls.

The candidate-wise gate is fixed as one shared scorer applied independently to each `(base, channel)` pair. The scorer emits one scalar per channel, then inactive scalars are set to negative infinity before the simplex softmax. It receives no channel identity or other channel embeddings.

A mandatory counterfactual test changes every masked-channel value while holding active channels fixed and requires bit-exact equality of candidate utilities, KEEP utility, fallback-correct logits, and all gate probabilities.

No formal model was fit and no outer label was opened before this correction. All channel masks, widths, losses, seeds, optimizer settings, thresholds, controls, and statistical gates remain unchanged.