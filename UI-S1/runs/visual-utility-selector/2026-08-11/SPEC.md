# Visual Utility Selector (VUS) Preregistration

Date frozen: 2026-08-11, before VUS sample generation, training, or evaluation.

Status: `FROZEN_BEFORE_VUS_RESULTS`

## 1. Motivation and boundary

Utility-LSA leaves substantial candidate oracle headroom (25.1--28.8 pp on Mind2Web and 14.1--17.6 pp on ScreenSpot-Pro) while improving CEV-A by only 0.25 pp on ScreenSpot-Pro and not improving Mind2Web. Its model sees only candidate-local structural statistics and scores each candidate independently. It cannot inspect the screenshot or task instruction, and a single global threshold must cover benchmark risks that differ by roughly threefold.

VUS is an independent post-Utility-LSA study. It does not amend, replace, or retroactively extend the frozen Utility-LSA protocol. All VUS choices below are frozen before VUS labels are materialized into training records or any VUS result is computed.

## 2. Hypothesis

A multimodal listwise selector that sees the task, screenshot, all 12 numbered candidate actions, and CEV-A's fallback can identify repairable CEV-A failures while avoiding harmful overrides better than structural Utility-LSA.

The cheapest falsification is nested inner-OOF evaluation: VUS must improve the same safe Step-SR objective over CEV-A and Utility-LSA before any outer-test result is accepted. Higher candidate-label accuracy alone is insufficient.

## 3. Inputs

- Frozen 12-candidate banks for `C_uni`, `C_cond`, `C_rand`, and `C_self`.
- Benchmarks: Mind2Web and ScreenSpot-Pro.
- Original screenshot and task/instruction only; Mind2Web receives task plus compact prior-action history.
- Each screenshot is overlaid with candidate labels `0`--`11`. Coincident labels are rendered with deterministic offsets and a legend containing normalized coordinates, action type, and parameter where applicable.
- Exact nested CEV-A behavior policy from Utility-LSA supplies the fallback index.
- Candidate correctness is used only for fold-local training reward and final held-out scoring. Target boxes, positive DOM candidates, and evaluator internals are never model inputs.

## 4. Model and optimization

- Base: locally retained Qwen3-VL-8B-Instruct, pinned by model index SHA-256 in the run manifest.
- Trainable parameters: LoRA on language attention/MLP projections; vision encoder frozen.
- Objective: supervised listwise utility policy with 12 output logits plus a `KEEP_CEV` logit.
- Primary target distribution gives equal mass to candidates with utility `+1`; when none exists, all mass is assigned to `KEEP_CEV`.
- Auxiliary loss predicts whether CEV-A is already correct. This is an explicit downside model, not benchmark test metadata.
- Candidate order is deterministically permuted from `(row_id, arm, epoch, seed)` during training and mapped back before evaluation.
- Mixed precision: bfloat16; gradient checkpointing enabled.
- Distributed execution: eight processes on GPU 0--7 via `torchrun`/FSDP or DeepSpeed ZeRO-2, without signalling or modifying protected PID 2274.
- Frozen search grid: LoRA rank `{16, 32}`, learning rate `{1e-5, 3e-5}`, epochs `{1, 2}`. Inner OOF selects one configuration globally.

## 5. Strict nested protocol

- Same existing five grouped outer folds; all arms of one underlying row remain together.
- For each outer fold, VUS configuration and safe thresholds are selected using predictions that are OOF over the four outer-dev folds.
- The final model is trained from the unchanged base on all four outer-dev folds and evaluated exactly once on the outer-test fold.
- No checkpoint, epoch, prompt, threshold, or hyperparameter may be selected using outer-test labels.
- All image overlays are deterministic and hash-audited. A generated training record containing target boxes, positive DOM nodes, or test labels in its prompt is a kill condition.

## 6. Safe policy

The direct prediction is the highest-probability candidate after mapping any duplicate coordinates back to a real candidate. `KEEP_CEV` preserves the exact fallback.

An override requires all conditions:

1. direct candidate differs from CEV-A;
2. `p(direct) - p(KEEP_CEV) >= threshold`;
3. predicted fallback-wrong probability is at least its threshold.

Thresholds are selected from inner-OOF quantiles. Unlike Utility-LSA, thresholds may be conditioned on benchmark and arm because the observed risk distributions differ materially. A hierarchical shrinkage rule backs cells with fewer than 200 OOF override opportunities toward the benchmark threshold. The exact rule is implemented and tested before outer-test evaluation.

Eligibility is unchanged in spirit: no inner-OOF cell may lose more than 0.5 MDE versus CEV-A and no benchmark equal-arm mean may lose more than 0.25 MDE. The selection objective is equal-benchmark/equal-arm standardized Step-SR delta.

## 7. Controls and gates

Controls:

- exact CEV-A;
- frozen Utility-LSA safe outputs;
- frozen correctness-LSA safe outputs;
- VUS direct without safe gate;
- text-and-structure-only VUS ablation;
- no-random-permutation ablation.

Primary gates:

- V1: all eight cells noninferior to CEV-A under the fixed MDE rule;
- V2: Mind2Web equal-arm 99% paired CI lower bound above zero;
- V3: ScreenSpot-Pro equal-arm 99% paired CI lower bound above zero;
- V4: equal-benchmark/equal-arm standardized 99% CI lower bound above zero versus Utility-LSA;
- V5: at least one benchmark gain is at least 1.0 pp and the other benchmark is noninferior.

Kill conditions:

- V-K1: any prompt or overlay leaks evaluator ground truth;
- V-K2: exact CEV-A fallback mismatch;
- V-K3: fewer than three outer folds select a finite override policy;
- V-K4: VUS fails to beat structural Utility-LSA in inner OOF before outer-test access;
- V-K5: any outer-test label influences training, checkpoint selection, prompt, calibration, or threshold selection.

## 8. Reporting

Report all eight cells, direct and safe outputs, wins/losses/override rates, five fold selections, 10,000 grouped paired bootstrap resamples with 99% percentile CIs, and exact model/data hashes. A result that fails V4 or V5 is not called the best aggregator even if one cell improves.
