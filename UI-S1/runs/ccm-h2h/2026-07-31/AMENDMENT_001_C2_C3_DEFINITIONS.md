# Amendment 001: Exact C2 and C3 Definitions

Date: 2026-07-31

Status: frozen after preregistration commit `89f492c` and before any C1-C3/H2/H4 result.

## C2 high-gap set

Recompute A5d-risk with the exact nested grouped folds from the upstream runner. For each pool, retain finite out-of-fold `S_gap` values. `high S_gap` is the inclusive 90th percentile by order statistic: sorted value at index `ceil(0.9*n)-1`. The diagnostic set contains rows with `S_gap >= threshold` on which the selected CCM candidate fails.

The all-model hard core is defined on the same deployable source set as the upstream pool. Report hard-core overlap within the diagnostic set, the pool hard-core base rate, enrichment ratio, and an exact hypergeometric upper-tail p-value. No threshold is selected from hard-core labels.

Prediction: high-gap failed rows are enriched in the hard core with one-sided `p < 0.01`.

## C3 error-conditional agreement mass

Use the original W1 deployable full-view candidate set and evaluator kernel. For candidate `i`, leave out its self-vote and compute `mass_i = mean_j!=i k(y_i,y_j)`. A row enters C3 only when at least one deployable candidate succeeds and at least one fails. The row collision statistic is the mean `mass_i` over failed candidates on that row.

Strata are fixed by benchmark and GT action:

- AndroidControl coordinate-bearing actions;
- AndroidControl string-bearing actions;
- AndroidControl parameterless actions;
- Mind2Web CLICK;
- Mind2Web SELECT+TYPE.

For each stratum, aggregation gain is A3 Step SR minus held-out A0 Step SR using upstream frozen row outputs recomputed under the same grouped folds. Report rows, mean error-conditional agreement mass, A0, A3, and gain. Test the preregistered law as Spearman correlation between collision mass and gain; prediction is negative. Because this is a post-P1 correction, its p-value is descriptive and does not retroactively rescue original P1.