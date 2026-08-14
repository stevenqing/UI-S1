# MASK Consensus-Occlusion Proposer Report

Date: 2026-08-14

Status: `MASK_STOPPED_M_K1_IDEAL_NEFF_GAIN_BELOW_MDE`

## Scope

MASK is an exploratory proposer study, independent from SPLIT. It does not change F1, Q1, `TRIVUS_NOT_PROMOTED`, or VUS-SR. The run stopped at the zero-GPU gate: no subset, GPU authorization, masked-model forward, or endpoint evaluation was created.

## Verifier closure

Sweeping the frozen SPLIT gate confirms the base-rate problem. At $g=0.25$, 1,187 rows enter the gate, but only 102 are M2-only positives: $\pi_0=8.59\%$ and $N/P=10.64$. Under the preregistered equal-variance Gaussian model, the best full-set net gain is:

| Hypothetical AUROC | Net gain |
| ---: | ---: |
| 0.75 | 0.0391 pp |
| 0.80 | 0.1919 pp |
| 0.85 | 0.5705 pp |
| 0.88 | 0.9557 pp |
| 0.90 | 1.2987 pp |

Thus the verifier route does not exceed the 0.70 pp MDE until roughly AUROC 0.88. The same qualitative constraint remains across the full frozen $g$ grid. See `VERIFIER_CONTOURS.pdf`.

## Effective-vote calibration and M-G1

All 4,095 nonempty subsets of the 12 C-uni source slots were evaluated with five-fold cross-fitting. The full-pool generalized $N_{\mathrm{eff}}$ is 1.5937. Under the deliberately favorable ideal $\kappa_{\mathrm{new}}=0$, three new votes add 1.3636 effective votes.

The within-benchmark monotone calibration predicts:

- density B3: **+0.538 pp**;
- F1 majority: **+0.219 pp**.

The maximum ideal prediction is +0.538 pp, below the preregistered 0.70 pp MDE. M-G1 fails and M-K1 stops the round before GPU. This calibration is benchmark-local and does not restore the previously rejected universal one-dimensional $N_{\mathrm{eff}}$ law.

## Base rates and mask control

Original C-uni density B3 accuracy is 63.69%; 574/1,581 rows are pool-wrong. M2 is correct on only 78/574 pool-wrong rows (13.59%). Fifty-five rows have only one C-uni mode under inherited $\tau^*$.

The deterministic equal-area empty mask is infeasible on 5/1,581 rows (0.32%), below the 15% control limit. M-K8 is not triggered; geometry/control feasibility is not the reason for stopping.

## Conclusion

> Even under the ideal assumption that three masked proposals are uncorrelated with the original pool, the frozen within-benchmark calibration predicts at most +0.538 pp, below the 0.70 pp MDE. MASK therefore stops before GPU and provides no evidence that consensus occlusion is a useful orthogonal proposer channel.

M-P2, M-P1, M-P3, M-P4, and M-P5 are all `NOT_RUN_M_K1_PRE_GPU_STOP`. Replacing the mask, radius, fill, proposer, aggregation, or adding a runtime gate would be a new study rather than a MASK rescue.
