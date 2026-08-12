# Amendment 001: Exact DELTA Training and Controls

Date: 2026-08-11

Timing: frozen after the result-free DELTA preregistration and before any DELTA model fit, prediction, or result.

## 1. Channel universe and variants

The model always instantiates five channel slots with shared channel encoder and identical parameter count:

1. `vus_binding`;
2. `global_semantic`;
3. `fine_local`;
4. `context_local`;
5. `random_placebo`.

Masked channels receive no gate probability. Fixed variants:

| Variant | Active channels |
| --- | --- |
| `FULL` | VUS, global, fine, context |
| `VUS_ONLY` | VUS |
| `VUS_GLOBAL` | VUS, global |
| `VUS_LOCAL` | VUS, fine, context |
| `RANDOM_PLACEBO` | VUS, global, random |

No variant is selected as the main model. `FULL` is primary; others are mandatory controls.

## 2. Features and architecture

Per channel/candidate features:

- centered label logit;
- log probability;
- probability;
- normalized within-row rank;
- row entropy;
- centered-logit difference from exact fallback;
- probability difference from exact fallback.

Candidate base features are frozen no-action fallback-pair structure, action category, benchmark/arm state, and fallback flag. Source/model/slot identity is prohibited.

All continuous base features are standardized from active model-training rows only. The seven channel-feature dimensions are standardized jointly across candidates and all five channel slots using model-training rows only; there is one shared mean/scale per feature dimension, not one scale per channel. Validation, OOF, and test rows use the frozen training statistics.

Architecture is exactly the main spec: shared channel encoder width 32, candidate-wise channel gate width 32, fused candidate width 64, two four-head Transformer layers, no positional embeddings, dropout 0.1, candidate/KEEP utility head, and fallback-correct head.

## 3. Training loss

For every batch, forward the original candidate order and a second deterministic candidate permutation. Restore the second output to original order. Consistency is the mean squared difference of:

- 12 candidate utility logits;
- KEEP utility logit;
- fallback-correct logit;
- 12-by-5 channel gate probabilities.

Loss weights remain:

- listwise repair-or-KEEP CE: 1.0;
- fallback-correct BCE: 0.5;
- permutation consistency: 0.1;
- expected U-GRPO utility: 0.1.

Optimizer:

- AdamW, learning rate `3e-4`, weight decay `1e-3`;
- batch size 256;
- exact split-weight normalization across accumulated micro-batches;
- one optimizer step per epoch;
- gradient norm 1.0;
- maximum 30 epochs;
- patience 5;
- minimum checkpoint improvement `1e-5`.

## 4. Triple-nested protocol

For each outer fold and trained variant:

1. among four development folds, each OOF holdout uses two model-training folds and one cyclic checkpoint-validation fold;
2. exact CEV behavior policy and reliability are fit on the same two model-training folds;
3. four OOF predictions select benchmark/arm safe thresholds;
4. final epoch is half-up median of four selected epochs;
5. final model trains on all four development folds;
6. atomically fsync all variant epochs/thresholds and channel hashes to `outer-k.pretest.json`;
7. only then open outer-fold labels and evaluate once.

Variants never share trained weights.

For paired capacity attribution, all trained variants within the same outer/inner context use the same initialization, candidate-permutation, row-order, and dropout seed. Inner seed is `20260811 + 1000*outer_fold + 10*holdout_fold`; final seed is `20260811 + 1000*outer_fold + 999`. Variants are reinitialized independently from the same seed and never warm-start from each other.

## 5. Safe policy

For trained variants:

- direct candidate: highest candidate utility logit;
- margin: direct candidate utility minus KEEP utility;
- downside score: `1 - sigmoid(fallback_correct_logit)`.

Use zero, infinity, and positive deciles on both axes. Select per benchmark/arm with the same eligibility as VUS-SR: cell loss at most 0.5 MDE and benchmark equal-arm loss at most 0.25 MDE, with benchmark backoff below 200 changed opportunities.

## 6. Fixed-average control

`FIXED_AVERAGE` uses no trained model:

- score is the arithmetic mean of per-row centered logits from VUS/global/fine/context;
- direct candidate is score argmax;
- margin is direct minus exact fallback score;
- downside score is constant one;
- thresholds are selected from development OOF rows with the same safe policy;
- no control or model selection uses outer-test labels.

## 7. Channel-dropout attribution

For each final FULL outer model, evaluate four additional outer-test predictions, each masking one real channel, without retraining or threshold recalibration. This is descriptive and cannot change the selected output.

DELTA-6 passes only when:

1. at least two real channels each receive mean gate mass >= 0.10 in at least four of five outer folds;
2. the identity of those channels agrees in at least four folds;
3. candidate permutation equivariance has maximum absolute restored-logit/gate error <= `1e-5` in tests.

## 8. Statistical gates

All method comparisons use paired grouped 10,000-resample bootstrap and 99% percentile intervals. Balanced effects average equal-arm benchmark effects after dividing by each benchmark MDE.

- DELTA-4: FULL minus VUS_ONLY balanced 99% lower bound > 0.
- DELTA-5: FULL minus RANDOM_PLACEBO balanced 99% lower bound > 0.
- `DELTA-K2` is the negation of DELTA-4.
- `DELTA-K3` is the negation of DELTA-5.

No architecture, channel mask, loss weight, optimizer, epoch cap, threshold grid, or attribution cutoff changes after this amendment.
