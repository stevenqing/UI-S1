# Source-Bias and Lineage-Normalized Aggregation Report

## Outcome

B1 passes at both scales. On incorrect B3 rows, GTA wins 489/574 times at 7B and 872/929 times at 72B, despite candidate-proportion expectations of 191.33 and 348.38. The standardized residuals are +26.36 and +35.49.

B4 does not support the stronger shared-proposer attribution at both scales. The view-0 GTA residual is not weaker than crop views at 7B, and 72B GTA within-lineage geometry is not significantly tighter than both alternatives. The defensible mechanism is a heterogeneous-pool aggregation effect. Candidate-count balancing nevertheless moves 72B B3 from 41.24% to 49.84%, a descriptive +8.60 pp gain.

## Nested B2

At 72B, nested lineage normalization reaches 70.59%, improving on B3 by +29.35 pp with 99% CI [+21.57, +35.78], and on M1 by +18.47 pp with CI [+12.95, +23.44]. It nearly realizes the Qwen3.5 best-single candidate headroom but remains -0.82 pp below it.

At 7B, nested lineage normalization reaches 61.99% and is worse than B3 by -1.71 pp, 99% CI [-3.09, -0.21]. The method therefore does not generalize across both frozen scales.

The preregistered B2 primary criterion fails because both scales were required. B-K4 triggers because the 72B nested result remains below best-single. B3x is not run by protocol.

## Claim boundary

The study establishes strong model-source voting bias in B3 and a large 72B correction from lineage normalization. It does not establish proposer-caused bias, a scale-general lineage-normalized method, or a result above best-single. Full-grid maxima remain descriptive.
