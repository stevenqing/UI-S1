# Scale-Up Gate Report

Date: 2026-08-02

Status: complete endpoint `B_CONTROLLED_7B_PLUS_SCALEUP_BOUNDARY`.

## G1 lineage gate

The three-model bare pass@3 is 80.20%. The minimum pairwise failure kappa is 0.460. G1 pass is `False`, lineage-concentrated is `False`, and the frozen G2 action is `RUN_G2_MARGINAL_GATE_STANDARD_THRESHOLD`.

Local bare scores and paper-only differences are reported in `MAIN_TABLE.md`. Anchor disagreement is treated as a reproducibility observation; no prompt or parser was retuned.

## G2 scale-up result

P1 GTA1-72B N8 M1 is 25.74%; P2 mixed N12 M1 is 49.15%. Their +23.40 pp difference is unequal-budget context, not an equal-compute claim. The proposal MDE is +2.35 pp. The frozen outcome is `BELOW_PAPER_MODEL_REFERENCE`.

The reported 70.4 and 73.1 values are independently source-verified but remain paper-only context, not same-environment controls, and never enter a row-level paired significance test. `REFERENCE_AUDIT.md` records the exact sources and protocol differences.

## 7B statistical close

Mixed N12 M1 reaches 63.82% versus V-only 60.40%: +3.42 pp, 99% CI [+1.41, +5.67] pp, one-sided p=9.999e-05. H3-native unchanged B3 moves from 60.09% to 63.63%: +3.54 pp, 99% CI [+1.27, +6.15] pp, p=9.999e-05.

All three Mixed-versus-bare comparisons and both N16 comparisons have positive 99% CI lower bounds. The N=2 H1 column moves to the appendix because M1/M2 collapse to the full-image prediction and M1 headroom capture is 0%.

## Scope corrections

The budget-decline claim uses V-only N=4 to N=16: B3 changes by -2.91 pp, supported by the negative X3 slope CI. H1 N=4 to N=10 is only a same-candidate-set rule comparison. Main-text MDE uses full/v1 exchangeable perturbations; v2-v4 are information deletion/deployment shifts.

The S-only GUI-RC sampling slope is -0.000285 per forward with 99% CI [-0.000789, 0.000203]. Because the CI crosses zero, the paper scope is **fixed-view allocation axis**, not a general single-model diversity axis.

## Paper disposition

UI-Zoomer X2 is excluded from the positive-result evidence chain and no X2 number enters this report. SafeGround, held-out pool ranking, and collision-floor evidence remain supporting diagnostics. No absolute open-source SOTA claim is made unless `system_SOTA_73_1_pass` is true.
