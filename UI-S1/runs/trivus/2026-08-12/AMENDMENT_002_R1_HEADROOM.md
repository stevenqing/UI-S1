# Amendment 002: R1 Complete-Bank Headroom Gate

Date: 2026-08-12

Timing: while result-blind R0 recovery is running, before `RECOVERY_MANIFEST.json` exists, before importing the AndroidControl evaluator in TriVUS, and before any recovered-bank candidate success or accuracy is computed.

An independent static audit during the same result-blind window required defense-in-depth revalidation. Before R1 can import scoring code, it now reparses all six actual lane artifacts, rechecks per-shard rows/bytes/SHA-256, exact 2,000-row stable-index coverage, reference order, provenance, and the combined ordered row-identity SHA-256 against the R0 manifest. This hardening does not alter active recovery workers, model inference, candidate content, or R1 statistics.

R1 is a one-time diagnostic on the complete six-lane 2,000-row Low/High bank. It is fail-closed: the recovery manifest must validate all lane identities/hashes and certify that R0 used no GT fields, scorer, evaluator, accuracy, or oracle computation before the R1 process imports scoring code or indexes any `gt_*` key.

For each setting and outer fold:

1. fit each source reliability on the other four episode-grouped folds;
2. form an exact action plurality among parsed UI-AGILE-7B, GUI-R1-7B, and UI-R1-E-3B candidates;
3. break action ties and choose the real candidate using descending development reliability, then frozen source order;
4. evaluate that candidate with the frozen AndroidControl action/coordinate/text scorer;
5. define candidate oracle success as any of the three real candidates succeeding.

R1 uses 10,000 paired episode-grouped bootstrap resamples within frozen folds and 99% percentile intervals. Blind AndroidControl selector inference is authorized only when `oracle - fold-local majority` is greater than 1.0 pp and its 99% CI lower bound is positive in both Low and High. Both conditions are required independently; no balanced average or favorable setting can compensate for failure in the other.

R1 does not select a model, prompt, candidate subset, fallback, or method hyperparameter. If it fails, TriVUS stops before new selector inference.