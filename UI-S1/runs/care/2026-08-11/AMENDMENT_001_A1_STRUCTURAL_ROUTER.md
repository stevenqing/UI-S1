# Amendment 001: Exact A1 Structural Router

Date: 2026-08-11

Timing: frozen after the diagnostic audit and before any CARE router fit, prediction, or result.

## Scope

A1 is a cheap viability test for adaptive acquisition. It does not use new VLM inference. It deliberately omits task/global-image embeddings; those belong to the full CARE router only if a stage1-only structural policy already captures nonzero counterfactual value.

## Inputs

Exactly the six public candidates shared across all four arms:

- parse state;
- action one-hot;
- normalized coordinate, coordinate-present, and out-of-frame state;
- parameter-present and log parameter length;
- action vote support and margin;
- candidate-to-peer distance statistics and medoid distance;
- coordinate support at radii 0.01/0.03/0.07/0.14;
- same-action coordinate support at the same radii;
- exact/soft parameter agreement;
- row coordinate dispersion;
- benchmark one-hot.

Prohibited: candidate 7--12 fields, stage-2 arm identity as input, source/model/slot identity, visual logits from a 12-candidate prompt, target/evaluator fields, and test labels.

## Target and objective

For each arm $a$:

$$
y_a=\mathbf 1[\exists c_i\in C_a:R_i=1].
$$

All four $y_a$ are available in the retained counterfactual bank. The model outputs four coverage logits. Loss is:

$$
\mathcal L=\mathcal L_{\mathrm{positive\ arm\ listwise}}+0.25\,\mathcal L_{\mathrm{four\ arm\ BCE}}.
$$

Listwise target mass is uniform over positive arms. All-fail rows contribute BCE only. All-pass rows have uniform listwise targets and therefore do not induce an arbitrary arm preference.

Each benchmark has equal total weight; each underlying row has equal weight within benchmark.

## Model

- shared six-candidate encoder, width 64;
- two Transformer encoder layers, four heads, no positional embeddings;
- mean pool plus benchmark state;
- four arm logits;
- dropout 0.1;
- AdamW, learning rate `3e-4`, weight decay `1e-3`;
- batch size 256;
- at most 50 epochs, patience 6, minimum validation improvement `1e-5`;
- candidate order deterministically permuted each epoch;
- gradient norm clipped at 1.0.

## Nested protocol

For each outer fold:

1. keep that fold's private-label file physically sealed;
2. on the four development folds, run four checkpoint fits, each training on three folds and validating on the fourth;
3. choose final epoch as the half-up median of four selected epochs;
4. choose one best-static arm per benchmark using all four development folds, with tie order `C_cond`, `C_uni`, `C_self`, `C_rand`;
5. train one final model on all four development folds for the frozen epoch count;
6. fsync `outer-k.pretest.json` containing epoch and static-arm selections;
7. only then open fold $k$ labels and evaluate once.

There is one model configuration, so no OOF architecture search.

## Primary outcome

Selected-arm pass@12 versus nested best-static-arm pass@12, paired by underlying row with the existing grouped bootstrap and 99% CI.

Secondary descriptive outcomes:

- selected-arm VUS-SR safe Step-SR versus the same nested best-static arm;
- arm choice frequencies;
- oracle route regret captured;
- benchmark and target-size strata.

## A1 gate

Proceed to CARE E0 when:

1. at least one benchmark has selected-arm pass@12 minus nested best-static 99% CI lower bound above zero;
2. the other benchmark point loss is smaller than its MDE and CI upper bound is nonnegative;
3. no benchmark's selected-arm VUS-SR safe Step-SR loses one MDE.

Failure closes learned routing for the current feature state. It does not authorize adding stage-2 leakage or tuning on the outer test.
