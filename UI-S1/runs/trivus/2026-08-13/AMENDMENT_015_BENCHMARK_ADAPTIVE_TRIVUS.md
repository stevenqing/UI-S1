# Amendment 015: Benchmark-Adaptive TriVUS

Date: 2026-08-13

Timing: after the completed formal TriVUS adjudication in commit `bdc4223bce04af257a62729391fdbb00f377eaad`. The prior outer labels and `TARGET_ONLY` outputs are already open. This amendment defines a post-hoc diagnostic and cannot alter the prior `TRIVUS_NOT_PROMOTED` outcome.

## 1. Method claim

The unified object is the candidate representation, variable-set selector architecture, nested OOF training procedure, safe-threshold algorithm, fallback policy, and held-out evaluation protocol. Model weights, standardizers, thresholds, and fallback baselines are benchmark-specific.

The canonical benchmark-adaptive policy is the existing formal `TARGET_ONLY` policy, composed without cross-benchmark prediction mixing from:

- `TARGET_ONLY_MIND2WEB`;
- `TARGET_ONLY_SCREENSPOT_PRO`;
- `TARGET_ONLY_ANDROIDCONTROL`.

`JOINT3` is an optional transfer comparison, not the benchmark-adaptive method and not a promotion requirement.

## 2. Locked exploratory inputs

The diagnostic reads only the five published formal outer results, their completion markers and pretest seals, the frozen public records, and the frozen primary and strongest baselines. It must reuse the exact grouped, fold-aware 10,000-replicate bootstrap implementation from the formal finalizer.

No model is retrained. No threshold is changed. No success bit is relabeled. The formal `TARGET_ONLY.safe` outputs are evaluated exactly as published.

## 3. Per-benchmark diagnostics

For each benchmark, compare `TARGET_ONLY.safe` separately against its frozen primary and strongest baselines.

- Cell safety: every cell's 99% confidence lower bound must exceed the negative benchmark MDE.
- Family improvement: the equal-cell family mean's 99% confidence lower bound must exceed zero.
- Benchmark-ready diagnostic: primary cell safety, strongest cell safety, and primary family improvement must all hold.

No three-family average may rescue a failed benchmark. No benchmark-ready result from this diagnostic is confirmatory.

## 4. Interpretation boundary

This analysis may determine whether the prior non-promotion is consistent with cross-benchmark negative transfer and may identify a benchmark-adaptive method candidate. It cannot promote TriVUS, replace the prior gates, or support a confirmatory claim because the outer labels were opened before this amendment.

A confirmatory evaluation requires a separately frozen protocol and untouched external or newly held-out labels.