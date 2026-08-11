# Post-result Correction 006: Fold-Sealed Label Access

Date: 2026-08-11

Timing: identified after the first formal VUS-SR adjudication and before git commit or paper use.

## Invalidated execution boundary

The first formal implementation indexed labels correctly by train/checkpoint/OOF/test fold, and no outer-test label entered a loss, checkpoint, configuration, or threshold calculation. However, `load_ranker_sources()` eagerly parsed the monolithic five-fold private-label JSONL before selection. The anchor adjudicator likewise joined all private labels before looping over outer folds.

Thus the numerical computation was fold-separated, but the process-level access boundary was not. Under the strict reading of V-K5, “loaded but not indexed” is still outer-test access before selection. The first compact results are invalidated pending a bit-identical hardened rerun.

## Preserved invalid artifacts

The invalidated compact artifacts are retained outside git at:

`/scratch/workspaceblobstore/visual-utility-selector/2026-08-11/INVALID_EAGER_TEST_LABEL_LOAD/`

| Artifact | SHA-256 |
| --- | --- |
| zero-shot anchor adjudication | `954d9e6c65d55ddda5c0ddce331362b986515c120c9b96cf7ad3804bc21d02d3` |
| set-ranker outer 0 | `07339f6311797b3c1ca7e3d7c3acf93468bc5e227738250937eb55a549f89773` |
| set-ranker outer 1 | `2f290ccaf9066aa4b5a63740387e03d3881b4056c45ea2f14f67b2b0b7ce8290` |
| set-ranker outer 2 | `e90fdc339d75b25b38ac972dfc2de57844d4b82f3e81062cac4313dd7a2f6695` |
| set-ranker outer 3 | `69d7368cef5632fa85021587563715996591bbd30d0937f46659bd132b4485f5` |
| set-ranker outer 4 | `0182c93ed3ea9416fd00ad8e2d9fb0bb47be9b80f7142963cb7f86cfee345f75` |
| set-ranker adjudication | `3bc252f2906a1a2a1d4cfdba90542edb1ab0325e90e875c87e0ebcd9571922fe` |
| set-ranker descriptive controls | `ee4b3965d226fabecf1fd45f5d5345e71e066cce121ac2c082b381b4a36cc280` |

## Hardened rerun contract

1. Partition private labels into five files before any model process starts.
2. An outer-fold process initially opens only the other four fold files.
3. It completes all inner fits, OOF configuration/threshold selection, final epoch selection, and final outer-development fit.
4. It atomically writes `outer-k.pretest.json`, including selected configuration, thresholds, epochs, and the hashes of the four opened development-label files.
5. Only after that file is fsynced may the process open `private_labels_fold-k.jsonl` and evaluate outer test once.
6. Anchor adjudication is likewise two phase: write all five pretest threshold selections using only development-label files, then open test-label files.
7. Hyperparameters, model seeds, features, losses, threshold grids, gates, blind visual logits, and bootstrap seeds remain unchanged.
8. Hardened rerun outputs must be bit-identical in selected configurations, final epochs, per-row safe/direct/fallback booleans, accuracy, and adjudicated gates. A mismatch stops the method claim for diagnosis; it does not authorize tuning.

This is a post-result implementation hardening, not a new preregistration or an independent confirmation.
