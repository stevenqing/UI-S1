# Difficulty-Conditioned Reallocation Report

Date: 2026-08-03

## Executive result

Difficulty-conditioned upward reallocation is not supported by the existing candidate bank. On the highest SafeGround-disagreement quintile, pass@N rises from 38.29% at N4 to 57.28% at N24, +18.99 pp, 99% CI [+14.11, +24.91] pp. B3 instead changes from 19.62% to 18.99%, -0.63 pp, 99% CI [-6.27, +5.76] pp.

R1 therefore fails and triggers R-K1. R2 budget reallocation and R3 conditional-proposal inference are cancelled exactly as preregistered. No S1-S4 result, random-budget control or new crop inference is fabricated after the failed gate.

This is the fifth direct collision-wall confirmation: additional candidates substantially increase oracle availability on difficult rows while unchanged B3 cannot realize the gain.

## Positive result: selective accuracy

SafeGround disagreement is highly useful for abstention even though it is not useful for deciding where to spend more of the existing fixed-view budget.

For Uniform Mixed N12, retaining the least-uncertain 90%, 80% and 70% yields B3 accuracies 69.06%, 74.60% and 79.02%, compared with 63.69% at full coverage. At 80% coverage, the gain is +10.91 pp; random rejection has mean 63.69% and 99% interval [+62.10, +65.35].

V-only N12 also benefits, but reaches only 67.48% at 80% coverage. Cross-lineage allocation thus improves both full-coverage grounding and the ranking of cases that should be deferred. The result supports a deployment workflow in which uncertain cases fall back to a human or a more expensive system.

R4 is a selective-prediction result, not evidence that the current uncertainty score can allocate additional fixed-view forwards effectively.

## 72B diagnostic

N3 already ruled out a global coordinate bug. R5 rejects the proposed tight-error-cluster explanation: 72B failed-candidate normalized pair distance is 0.1539 versus 0.1137 at 7B, with paired 72B-minus-7B delta +0.0616 and positive 99% CI.

However, B3 shows severe source bias. Among 929 wrong 72B B3 rows, the selected candidate comes from GTA1 on 872 rows, UI-Venus on 52 and Qwen3.5 on only 5. Winner-cluster model composition is highly nonuniform (`p=5.66e-123`).

The supported diagnosis is therefore model-source/coverage bias in B3 selection, not unusually tight strong-model errors.

## Preserved contribution

The paper retains:

1. equal-compute cross-lineage gains of +3.42 pp M1 and +3.54 pp unchanged B3 at 7B N12;
2. fixed-view budget-slope sign reversal;
3. weaker-model complementarity;
4. significant N8 CALA gains at 7B and 72B;
5. selective accuracy as a new deployment-facing positive result;
6. repeated evidence that oracle candidate headroom and final aggregation accuracy are distinct bottlenecks.

No R3 inference is launched, so this run consumes zero new model forwards.
