# Benchmark-Adaptive Safe Utility Selection

## Objective

Use one method across GUI benchmarks without sharing fitted model parameters across benchmarks. Each benchmark trains and calibrates its own candidate ranker and override gate against its own strongest frozen fallback.

## Model

The method has three benchmark-specific stages.

1. A contextual candidate-success verifier is trained on every valid candidate label using row-normalized BCE and within-row positive-versus-negative ranking loss.
2. The highest-scoring candidate becomes the direct candidate.
3. An incremental-utility head predicts whether overriding the strongest fallback is beneficial.

For a direct candidate outcome `d` and strongest fallback outcome `b`, the override target is:

- `win`: `d = 1, b = 0`;
- `loss`: `d = 0, b = 1`;
- `tie`: `d = b`.

The predicted incremental utility is `P(win) - P(loss)`. The method overrides only when this value exceeds a benchmark-calibrated minimum and `P(loss)` is below a benchmark-calibrated maximum. Otherwise it returns the strongest fallback.

The old KEEP-centric target is retained only as a control. It is not the primary candidate-learning objective because fallback-correct rows otherwise provide no direct supervision for alternative candidate correctness.

## Leakage control

The override head must be trained only on base-ranker OOF predictions. It may not consume in-sample direct choices. Thresholds must be selected from development OOF rows and sealed before outer labels are opened.

Each outer fold therefore contains:

1. benchmark-specific base-ranker OOF training;
2. benchmark-specific override-head OOF training on cross-fitted base predictions;
3. safe threshold selection against the strongest fallback;
4. final base-ranker and override-head fitting on outer development folds;
5. artifact reload and pretest seal;
6. one-time outer-label access.

## Evaluation

Each benchmark is adjudicated independently. Required confirmatory gates are:

- every cell is non-inferior to the strongest fallback at the frozen benchmark MDE;
- the equal-cell benchmark mean has a positive 99% confidence lower bound;
- no cross-benchmark average can rescue a failed benchmark.

The 2026-08-13 implementation and diagnostic are exploratory because the existing outer labels were opened before this specification. A promotion claim requires untouched labels under a separate authorization.