# Amendment 003: Exact Noninferiority Test

Date: 2026-08-11

Timing: frozen during result-free implementation audit; before any formal outer-fold fit, outer-label access, or DELTA result.

The phrase "noninferior under the 0.70 pp MDE" was underspecified and nearby studies used incompatible heuristic rules. DELTA-2 therefore uses the standard confidence-bound test for every ScreenSpot-Pro arm:

$$
\operatorname{lower}_{99\%}(\Delta_{\mathrm{FULL}-\mathrm{VUS}}) > -0.007.
$$

Neither a favorable point estimate nor a confidence interval that merely crosses zero can substitute for this condition. This definition is fixed before results and tested with synthetic boundary cases.

The implementation also validates the frozen YAML against the executable channel order, variant masks, loss weights, optimizer-step count, threshold policy, nested split contract, and statistical protocol before every formal fold and adjudication.