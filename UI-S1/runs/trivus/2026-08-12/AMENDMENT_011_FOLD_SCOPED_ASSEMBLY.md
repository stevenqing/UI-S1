# Amendment 011: Fold-Scoped TriVUS Assembly

Date: 2026-08-12

Timing: after source-free data-primitives commit `2a15152` and exact fallback-context seal commit `9e5b22b`, before any real private-label assembly, standardizer fit on real rows, optimizer step, checkpoint selection, or model result.

## 1. Access boundary

Public candidates and blind visual predictions are loaded from locked hashes and contain no success labels. A context slice is selected by outer fold, role, and optional OOF holdout. Private labels are then opened only from an explicit requested-fold tuple. The private loader requires exact expected key equality for those folds and never preloads the other folds.

For an inner phase, the only legal requested fold sets are:

- the exact two model-training folds;
- the one checkpoint fold;
- the one OOF holdout fold.

For a final phase, the legal sets are the four outer-development folds or the one outer-test fold. A phase context is available only on its preregistered applied folds.

## 2. Exact input schemas

VUS public rows have exactly schema version, sample/benchmark/arm/row/fold/group metadata, image path/hash, instruction, history, and candidates. AndroidControl uses setting instead of arm. Candidates have exactly action, coordinate, parameter, and parse-ok.

VUS and AndroidControl blind prediction schemas are separately exact and must match public sample key, benchmark, cell, row, fold, group, image hash, K-wide display permutation/logits/probabilities, and frozen model-index hash. Extra source/model/target/success fields are rejected.

## 3. Context slice

The 391,524-row context bank is streamed and not expanded into a global feature tensor. The selector requires exact context key syntax, expected fit folds, applicable public folds, public sample identity, and fallback range K. The selected key set must exactly equal the public rows in the requested folds.

## 4. Assembly and weights

`assemble_data` creates validated source-free `TriVUSData` with zero row weights. It pads K=3 rows to 12, restores visual evidence to public candidate order, derives targets from the requested physical labels, and preserves context/sample/family/cell/fold/group metadata.

`with_model_weights` is a separate explicit operation. It supports exactly:

- JOINT3 / NO_VISUAL / RANDOM_ID_PLACEBO: all three families;
- TARGET_ONLY: exactly one declared family;
- JOINT2_NO_ANDROID: Mind2Web and ScreenSpot-Pro.

Excluded-family rows receive zero weight. Each included family and cell follows Amendment 010. Standardization is fit only after model weights are assigned and only on model-training rows.

## 5. Execution boundary

The initial implementation and tests use synthetic temporary public, prediction, context, and physical fold files. They do not open real private labels. A separately committed one-time authorization is required for one real inner-development smoke. That smoke may validate schemas, counts, targets, weights, standardization, and a forward/loss pass with no optimizer step. Formal nested training remains unauthorized.