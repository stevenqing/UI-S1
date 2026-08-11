# GTA1 / MVP Reference Audit for Utility-LSA

Date: 2026-08-11

## GTA1

Verified sources:

- `Yan98/GTA1@81f0c4c2997cf6ad696d42f47c5480f4c856b73a`
- `Yan98/GTA1/src/grpo_grounding.py`
- `Yan98/GTA1/src/trainer/grpo_trainer.py`
- `Yan98/GTA1/src/trainer/grpo_config.py`
- GTA1 Hugging Face article, “GRPO for GUI Grounding Done Right”

Frozen semantics relevant here:

- `num_generations=8`;
- binary click reward: predicted point inside the target bbox;
- optional format reward, while the published recipe invokes `reward_funcs=accuracy`;
- group advantages `(reward - group_mean)/(group_std + 1e-4)`;
- default sampling temperature around 0.9 in config, generation code uses sampling;
- GRPO optimizes the grounding policy's completion log probabilities;
- the blog reports that click-based reward is sufficient, KL can be disabled, and all-correct/all-incorrect groups can make the reward signal vanish.

GTA1 does **not** train the MVP aggregation rule. It trains the model that emits grounding coordinates. Its agent-level test-time scaling also samples action proposals and uses a judge model, which is distinct from the static grounding aggregator studied here.

## MVP

Pinned source:

```text
ZJUSCL/MVP@988ff3c61b9f7632d780ae27c83260de75b3c95f
```

Verified aggregation semantics for GTA1 ScreenSpot-Pro:

1. candidates remain in view order;
2. deterministic greedy complete-link grouping;
3. every member must be within 14 pixels on both axes of every existing group member;
4. group size is the primary score;
5. mean AGVP coverage breaks group ties;
6. highest-coverage real prediction in the winning group is returned.

The paper's centroid prose and official code's real-member output differ; this project uses the official-code behavior when naming MVP.

Main MVP is a training-free inference-time aggregation and view-generation method. However, Section 8 evaluates a separate learned coordinate selector:

- roughly 20k high-resolution/small-target training samples;
- 2–4 candidate points rendered as numbered red dots, one inside the target and the others outside;
- Qwen3VL-4B-Instruct predicts the correct point label;
- binary reward 1 for the correct label, otherwise 0;
- GRPO with 8 rollouts per group, gradient accumulation 32, 170 optimization steps on 8 A6000 GPUs;
- GTA1-7B candidates: base selector 60.5%, trained selector 62.8%, clustering 61.7%;
- Qwen3VL-8B candidates: base selector 62.7%, trained selector 65.3%, clustering 65.5%.

The selector improves over its untrained base but does not consistently beat clustering across candidate generators. The paper therefore retains clustering as the main robust aggregator.

## Transfer to Utility-LSA

Utility-LSA intentionally combines only the transferable principles:

- from GTA1: evaluator-aligned group reward and within-group relative advantage;
- from MVP main method: complete-link/support geometry and real-candidate output;
- from MVP selector ablation: candidate-label GRPO is feasible but absolute selection reward does not consistently beat clustering;
- from this project: utility measured relative to a strong cross-fitted CEV-A behavior policy.

The resulting method is an offline group-relative utility regressor over fixed heterogeneous candidates. Unlike MVP's rendered-dot VLM selector, it trains directly on override utility relative to CEV-A and uses structural features. It is not on-policy GRPO, does not update a VLM, and does not reproduce GTA1's policy-gradient optimization.

AGVP coverage is unavailable in a comparable form on Mind2Web and is not fabricated. Cross-fitted source reliability is reported only as a generic quality feature.
