# Amendment 003: Visual-Logit Set Ranker Before Full LoRA

Date: 2026-08-11

Timing: frozen while clean fallback-agnostic anchor inference was still running, before private labels were joined to clean anchor predictions and before any VUS anchor or learned result was computed.

## Motivation

Full five-by-four nested Qwen3-VL LoRA selection is expensive. The clean blind anchor already extracts screenshot/task-conditioned candidate evidence as 12 one-step logits. Utility-LSA showed that independent candidate regression and one global threshold are too weak; it did not test a permutation-equivariant listwise model with visual evidence and benchmark/arm risk conditioning.

## Stage VUS-SR

Before full VLM LoRA, train a small GPU set ranker over frozen clean visual logits plus frozen structural candidate features.

Inputs per candidate:

- clean Qwen3-VL candidate logit, probability, rank, entropy, and margin to the exact nested CEV fallback candidate;
- frozen no-action LSA candidate/fallback pair features;
- candidate action type and parameter-presence state;
- benchmark and arm indicators, allowed here because they parameterize observed downside risk rather than candidate source identity;
- one fallback flag.

Prohibited inputs remain source/model/slot identity, target boxes, evaluator internals, raw responses, and test labels.

Architecture:

- shared candidate encoder, width 64;
- two Transformer encoder layers, four heads, no candidate positional embeddings;
- one learned KEEP token initialized from the fallback candidate state;
- candidate and KEEP utility logits plus one fallback-correct auxiliary logit;
- dropout 0.1.

Frozen configurations:

| ID | Objective | Learning rate | Weight decay | Epochs |
| --- | --- | ---: | ---: | ---: |
| S1 | soft listwise repair/KEEP CE | 3e-4 | 1e-3 | 30 |
| S2 | S1 + 0.5 fallback-correct BCE | 3e-4 | 1e-3 | 30 |
| S3 | S2 + 0.25 negative expected U-GRPO utility | 1e-4 | 1e-3 | 50 |

Training uses early stopping only inside the inner-training folds: reserve the cyclic first training fold as checkpoint-validation and fit on the remaining training folds. Patience is five epochs, maximum epoch as above. The inner holdout remains untouched for configuration and safe-threshold selection. Final outer models choose the median selected epoch across four inner fits and train on all four outer-development folds for that many epochs.

Loss weighting gives equal total mass to each benchmark, then each underlying row, then each active arm. Candidate order is deterministically permuted per epoch. Soft target mass is uniform across `+1` utility candidates; if none exists, mass is on KEEP. The auxiliary target is exact nested fallback correctness. U-GRPO uses sample standard deviation plus `1e-4`, matching Utility-LSA.

## Nested behavior policy

For each `(outer fold, inner holdout)` model, one exact CEV policy is fitted on the three inner-training folds and applied to both training and holdout rows, matching Utility-LSA. Final outer training and test use the frozen outer-fold CEV policy fitted on the four development folds. Any outer-test CEV correctness mismatch is V-K2.

## Selection and gates

Inner OOF selects S1/S2/S3 and benchmark/arm safe thresholds under the VUS eligibility constraints. Outer-test labels are opened once. VUS-SR is promoted over full LoRA when:

- all eight cells are noninferior to CEV-A under the fixed MDE rule;
- at least one benchmark improves by at least 1.0 pp and the other is noninferior;
- equal-benchmark/equal-arm standardized 99% paired CI is positive versus Utility-LSA.

If these hold, VUS-SR is the learned method candidate and full LoRA is unnecessary. If clean anchor passes A1/A2 but VUS-SR misses promotion, execute the preregistered full LoRA stage. If clean anchor fails A1/A2, neither set-ranker nor LoRA training is authorized.
