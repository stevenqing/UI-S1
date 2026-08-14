# ORTH Amendment 002: Arm 3 identifiability

Date: 2026-08-14
Timing: before Arm 3 implementation and before any Arm 3 table is computed.

The original scoping text requested fused accuracy from only a hypothetical channel's marginal accuracy and error kappa with the visual pool. Those two moments determine the 2-by-2 joint correctness table, but they do not determine candidate identity, a row-level likelihood ratio, sensitivity/specificity with respect to a common binary latent decision, or what a log-odds fusion rule observes. Therefore Bayes-fused grounding accuracy is not identified.

Arm 3 is restricted to quantities identified by the two moments:

1. Project the requested $(\mathrm{accuracy},\kappa)$ onto the feasible 2-by-2 error-coupling interval and report requested versus achieved kappa.
2. Report both-correct, visual-only-correct, new-only-correct, and both-wrong probabilities.
3. Report fixed visual accuracy, forced-new accuracy, random choice on disagreement, and the oracle selector upper bound $1-P(\text{both wrong})$.
4. Report weight dominance for visual weights 12, 1.5936767669403409, and 1 against a unit-weight new channel. Without row-level confidence, weights greater than one always retain visual on disagreement; equal weights correspond only to an explicitly random 50/50 tie sensitivity, not Bayes fusion.
5. The Gaussian-copula construction may verify attainable couplings over 100 seeds, but it must not be labelled fused performance.

The resulting 2-D table is design headroom. A later confirmatory fusion protocol must freeze a common candidate space and calibrated per-candidate channel likelihoods before log-odds addition can be evaluated.
