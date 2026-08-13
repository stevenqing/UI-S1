# Amendment 001: SSPro Stratum Feasibility

Date: 2026-08-14

Timing: after preregistration commit `3a36e0d`, input-lock commit `35955af`, CLICK-scope commit `193d810`, and anchor commit `ba8c6b5`; before any $\hat p$, $\hat q_{\max}$, margin, contamination, or $\tau$ sweep computation.

The original specification simultaneously fixed four ScreenSpot-Pro strata over 1,581 rows and required every reported stratum to contain at least 400 rows. Four strata cannot all satisfy that threshold because $1581 < 4\times400$. This makes G-P1 structurally unreportable regardless of observed values and conflicts with G-K1.

The correction preserves four ScreenSpot-Pro strata and sets its G-K5 minimum to 395 rows. Under deterministic quantile allocation the expected counts are 396/395/395/395 before ties. Mind2Web retains the 400-row minimum and its already locked four-stratum decision from 1,774 CLICK rows.

No strata are merged to rescue a result. Ties at a quantile boundary must be assigned by stable row identity to preserve the fixed counts. The prediction, confidence level, primary-test designation, and all other kill conditions are unchanged.

The Assumption A2 cross-reference is also corrected from G-P6 to G-P8. G-P8 is the test of the kappa-as-ICC quantitative prediction; G-P6 is the endpoint convergence test.