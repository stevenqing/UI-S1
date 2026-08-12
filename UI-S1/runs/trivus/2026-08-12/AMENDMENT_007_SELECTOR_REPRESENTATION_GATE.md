# Amendment 007: Blind Selector Representation Gate

Date: 2026-08-12

Timing: after the fold-sealed private-label manifest was committed as `0c74d064913c9b91e5c570eb4b366dfde0b0d08d`, before any blind-selector accuracy, AUROC, random-control result, TriVUS data fit, or model training.

This one-time gate determines whether the locked AndroidControl blind logits contain enough utility signal to justify nested TriVUS training.

Execution is additionally sealed in two commits. First, the complete gate implementation is committed while no authorization file exists, so the executable path fails closed. A later authorization record must name that implementation commit and lock SHA-256 values for this amendment, the gate config, gate code, and tests. The gate verifies that commit ancestry and all file hashes before loading private labels or computing metrics.

Before label access, the gate rehashes the raw public and prediction JSONLs against their manifests, verifies the manifests declare those same hashes, audits all 4,000 public schemas/images, and reruns the full blind prediction validator including prompt, overlay, permutation, probability, argmax, model, image, and public identity checks.

## Frozen rows and policies

For each setting and outer fold, reconstruct the fold-local majority fallback exactly from the public candidate actions:

1. use the other four folds to estimate each frozen candidate source's candidate-success reliability;
2. choose exact action plurality among parsed candidates;
3. break action ties and choose a real candidate by descending development reliability, then frozen canonical source order;
4. convert the selected canonical source index into the source-neutral public candidate index using the committed public-order permutation.

Blind direct is the locked A--C probability argmax. Hash-random is `SHA256(sample_key/20260812/representation-random) mod 3`, independent of labels and model scores.

## Diagnostics

For Low and High separately:

- direct minus fold-local majority paired episode-grouped 10,000-bootstrap effect and 99% CI;
- direct minus hash-random paired episode-grouped 10,000-bootstrap effect and 99% CI;
- fallback-relative repair AUROC, where a candidate is positive iff it succeeds and the fold-local majority fallback fails; selector probability is restored from display order to public candidate order.

## Gates

- `RG-S`: direct minus hash-random 99% CI lower bound is positive in both Low and High;
- `RG-A1`: at least one setting has direct-minus-majority 99% CI lower bound positive, the other setting has lower bound greater than `-0.01`, and the equal-setting point mean is positive;
- `RG-A2`: fallback-relative repair AUROC is at least 0.55 in both settings.

Proceed to TriVUS data/training implementation only when `RG-S` passes and either `RG-A1` or `RG-A2` passes. No threshold, safe policy, prompt, candidate set, or model is selected by this diagnostic. Failure stops TriVUS before training.