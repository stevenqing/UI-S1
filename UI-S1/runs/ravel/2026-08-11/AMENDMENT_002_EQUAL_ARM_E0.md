# Amendment 002: Equal-Arm E0 Diagnostic Aggregation

Date: 2026-08-11

Timing: frozen while fine/context blind logits were incomplete and before any RAVEL evidence labels were opened or E0 metric computed.

The E0 protocol names all four arms as mandatory cells but did not state whether diagnostic candidate AUROC and recall pool all arm-candidates or weight arms equally. Because arms can repeat candidates differently, pooled candidate counts can implicitly weight one acquisition policy more.

Frozen primary aggregation:

1. Compute utility-positive candidate AUROC separately for C-uni/C-cond/C-rand/C-self.
2. Primary benchmark AUROC is the arithmetic mean of four arm AUROCs.
3. Compute direct accuracy, unique-correct recall, and smallest-target-quartile recall separately per arm; report their equal-arm means as primary.
4. Retain pooled candidate/row metrics as descriptive diagnostics only.
5. E0's `+0.03 / −0.01` gate and local-vs-random condition use equal-arm metrics.

This matches VUS-SR's equal-arm deployment interpretation and does not change any logits, crops, prompts, or labels.
