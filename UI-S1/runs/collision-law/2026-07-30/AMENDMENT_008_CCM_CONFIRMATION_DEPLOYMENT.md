# Preregistration Amendment 008: CCM Confirmation Deployment

Date: 2026-07-31

Frozen after Amendment 007 and all W1/W2 discovery results, but before any W4 inference. This amendment closes only the discovery-to-confirmation deployment details that Amendment 007 left implicit. It does not change the A5 estimator, candidate classes, pair types, bins, smoothing, backoff, K4, or success criteria.

## Confirmation source sets

Each AndroidControl-Curated setting uses exactly the corresponding W1 deployable source set:

- Low: GUI-R1-3B, GUI-R1-7B, UI-AGILE-3B, UI-AGILE-7B, and UI-R1-E-3B;
- High: GUI-R1-3B, GUI-R1-7B, UI-AGILE-3B, and UI-AGILE-7B.

UI-R1-E-3B is excluded from High CCM evidence and candidacy because it failed the frozen discovery deployability band. Its W4 inference remains useful for the preregistered A0-A4 robustness table but cannot enter CCM confirmation.

The fixed best source is selected by full discovery Step SR before W4: UI-AGILE-3B for Low and UI-AGILE-7B for High. W4 labels cannot change either source.

## Final calibration

Low and High are calibrated separately.

1. For each of the five frozen grouped discovery folds, fit A5d nine-cell LR tables and source MAP priors on the other four folds.
2. Score the held-out fold once, retaining `S_gap` between A5d and the fixed best-source candidate. These are out-of-fold discovery scores.
3. Pool the five held-out folds. Candidate thresholds are all observed nonnegative out-of-fold `S_gap` values plus infinity.
4. Select the smallest threshold whose pooled out-of-fold Step SR is at least the pooled fixed-best-source Step SR. Ties are inclusive. Infinity is feasible by construction.
5. Refit A5d LR tables and source priors once on all discovery rows. This full-discovery calibration plus the frozen out-of-fold threshold is applied to every W4 row.

No W4 label, corrected action, candidate action, bbox, or metric result enters these five steps. W4 predictions enter only after the final calibration and threshold are serialized.

## Confirmation outputs

For each setting, report frozen best-source and CCM Step SR, paired wins/losses, exact McNemar p-values, override rate, conditional override Step SR, `S_gap` correctness AUROC as a diagnostic, and LR backoff counts. Confirmation succeeds directionally when CCM-minus-best-source has the same sign as discovery A5d-risk. The original Amendment 007 full-method criterion remains unmet because K4 already triggered; W4 cannot reverse K4.

## Mind2Web confirmation availability

Public asset audit found the official `osunlp/Multimodal-Mind2Web` dataset at revision `1b4c6a8cf9f77b7a5e0d641959935c80c4a05889`, but no official `MM-Mind2Web-v2` corrected-label release, auditable label diff, identity mapping, or revised evaluator. Search results named `v2` are unrelated third-party reward-model or agent-training datasets. They are not accepted as confirmation labels. Mind2Web A5 results therefore remain discovery-stage unless a versioned official correction set becomes available.