# Amendment 012: No-Optimizer Real-Data Smoke

Date: 2026-08-12

Timing: after fold-scoped assembly commit `1a1ea5e8b4a6c68c5d3feed744de57d5fccbef0e`, before any real private-label assembly, optimizer construction, backward pass, checkpoint selection, thresholds, outer labels, or model result.

## 1. Fixed phase

The smoke runs exactly one inner phase:

- outer fold 0;
- OOF holdout fold 1;
- checkpoint fold 2;
- model-training folds 3 and 4.

Expected public row counts are 7,428 model-training rows, 3,838 checkpoint rows, and 3,666 OOF rows. Each subset opens only its exact VUS and Android physical label folds through a validated `PhaseContext`.

## 2. Allowed checks

The smoke may:

1. validate all public, blind-prediction, context-manifest, and physical-label schemas/hashes;
2. assemble source-free K=3/12 data for the three subsets;
3. assign JOINT3 family/cell weights to model-training rows only;
4. fit a JOINT3 standardizer on model-training rows only;
5. apply the frozen standardizer to checkpoint and OOF rows;
6. initialize the frozen TriVUS model with seed 20260822;
7. run one deterministic no-gradient forward/loss check on at most 64 positive-weight training rows.

## 3. Prohibitions

The runner may not construct an optimizer, call backward, mutate model parameters, select an epoch/configuration/threshold, open outer fold 0 labels, report numeric loss, accuracy, success rate, active-row count, target distribution, or any candidate-level private value.

The output contains only public row counts, physical opened-fold provenance, boolean contract checks, implementation/authorization hashes, and explicit false flags for optimizer, backward, performance metrics, and training.

## 4. One-time authorization

The runner and tests must be committed result-free. A separate committed authorization binds their exact Git blobs and a fresh nonce. The nonce is consumed before any private label is opened. A failed run requires a new implementation/authorization boundary.

## 5. Execution boundary

Passing this smoke authorizes implementation of the nested runner, but not formal fits. Formal training still requires the complete runner, threshold code, pretest seal, final adjudicator, result-free tests, and a separate authorization.