# TriVUS Exact Fallback Contexts

Date: 2026-08-12

## Outcome

`PASS_TRIVUS_EXACT_FALLBACK_CONTEXTS`

- 18,644 public samples;
- 391,524 minimal context records;
- five final and sixteen inner contexts per sample;
- 25 exact split records: five final plus twenty inner;
- 14,644 frozen VUS-SR final-index anchors checked;
- zero final-index mismatches;
- context SHA-256 `470b41f623bca5bcf740ce4a599258d27432a4e2946751559a525e3c4082ef7b`;
- manifest SHA-256 `8b391a5a381714e4dec777a93c59871359bb195368c9c3d66198d8cc3b51562a`.

Each context contains only schema version, context/sample key, outer fold, role, holdout fold, fit folds, and fallback index. It contains no success bit, target, source/model/lineage/slot identity, reliability, policy configuration, configuration score, or aggregate performance value.

Each manifest split records the exact VUS-label, Android-label, and Mind2Web-scale physical fold hashes opened for that fit. Inner contexts use two model-training folds, one checkpoint fold, and one OOF holdout. Final contexts use all four outer-development folds.

## Execution corrections

Three fail-closed corrections preceded the successful artifact:

1. nested workspace paths were resolved relative to the actual Git root;
2. frozen zero-dimension Mind2Web target scales were accepted as finite nonnegative values;
3. normalized ScreenSpot coordinates were restored to their exact integer pixel grid before the 14-pixel CEV threshold.

Each failed authorization was permanently consumed and retained. No failed attempt published a destination context directory. No performance metric was computed and no model fit was started.

## Boundary

This is a preprocessing artifact, not a model result. It authorizes neither unified feature assembly nor training. AndroidControl remains the paired 2,000-row Low/High sample; Mind2Web/ScreenSpot-Pro comparisons remain success-bit-only against frozen VUS-SR.