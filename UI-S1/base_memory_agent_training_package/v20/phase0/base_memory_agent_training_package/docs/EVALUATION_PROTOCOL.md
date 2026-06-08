# Evaluation Protocol

This package uses two separate evaluations. Do not conflate them.

## 1. GT-History Coverage Diagnostic

Purpose: measure whether a trained proposal agent adds useful candidates when it is not punished for earlier trajectory mistakes.

Procedure:

1. Export/merge the full-SFT checkpoint to a vLLM-loadable model path.
2. Run the official v13 template evaluator with `GT_HISTORY=1` and full 1000 episodes.
3. Add the trained source to the current candidate pool.
4. Recompute oracle using `v20/phase0/scripts/analyze_oracle_reranker.py`.

Metric:

```text
candidate-pool oracle TSR
```

Current reference:

```text
prompt-specialized expanded pool oracle = 32.7%
```

Target:

```text
>= 35.0%
```

## 2. Autoregressive Deployment

Purpose: evaluate the actual system.

Procedure:

1. At step t, all memory agents condition on the selected/executed history so far, not GT history.
2. Each agent proposes one candidate action.
3. Selector/aggregator chooses one candidate.
4. The selected action is executed and becomes part of the next history.
5. Evaluation uses official stop-on-error trajectory metrics.

Metrics:

- TSR
- avg_progress
- step_sr
- mean_reward
- source selection counts
- first-error decomposition

## Acceptance Criteria

A trained memory-agent family is useful only if:

1. GT-history oracle coverage improves over 32.7%.
2. Autoregressive selector deployment improves over the current locked selector/controller baseline.

Standalone agent TSR is diagnostic only and should not be the main claim.
