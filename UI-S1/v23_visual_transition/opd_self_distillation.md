# V23 OPD Self-Distillation Notes

This note consolidates the V23 offline experiments and reframes the next step as
OPD self-distillation: use the current best model as the teacher, preserve its
behavior where it is already good, and apply repair signal only on states where
the teacher demonstrably fails.

Here OPD means offline policy distillation. It is offline because every state is
a fixed GT screenshot, and it is self-distillation because the teacher is the
best V23 checkpoint rather than a separate stronger model.

## Current Best

The current reporting checkpoint is:

```text
checkpoints/v23_where_what_route_only_from_step5_short_fullsave/epoch-0_step-10
```

Full balanced 1000 result:

```text
TSR      22.3%
StepSR   69.07%
Progress 35.21%
```

First-200 slice result:

```text
TSR      18.5%
StepSR   67.72%
Progress 32.48%
```

This checkpoint starts from WHAT/WHERE routed SFT step-5 and then trains only the
route weights for 10 extra steps. It is the only V23 branch so far that improved
the full balanced 1000 result over the full SFT baseline.

## Evaluation Geometry

GUI-360 balanced eval uses GT screenshots with `stop_on_error`. This has two
important consequences:

1. The model is not being tested on online recovery after entering an off-GT GUI
   state.
2. TSR is mostly about delaying or removing the first local action error.

The local decision is approximately:

```text
p(action_t | GT_screen_t, goal, action_history_<t)
```

Since the text error-horizon probe found that isolated text-history errors do not
cause meaningful damage under GT-screen rescue, the main lever is not better
textual error propagation. The main lever is better local visual-action binding
on the current GT screen.

Related text-space diagnostic:

```text
outputs/text_error_horizon_probe/qwen35_pilot300_action_only/text_error_horizon_report.md
```

That probe showed a near-zero causal text-history horizon under GT-screen rescue.
Natural curves by `prefix_error_count` looked large, but controlled injection of
one to three real wrong actions stayed nearly flat. The interpretation is that
long-horizon offline failures are not mainly caused by textual action-history
error propagation. They are better explained by local hard-state correlation,
task coupling, latent state binding, and in online settings off-GT screen drift.

## What Improved

Compared with the full SFT baseline, current best full 1000 improves:

```text
baseline full SFT:       21.7% TSR / 68.22% StepSR / 34.77% progress
WHAT/WHERE routed step5: 22.1% TSR / 68.49% StepSR / 35.02% progress
route-only step10:       22.3% TSR / 69.07% StepSR / 35.21% progress
```

The gain is real but small. On the full 1000 comparison, current best fixes 11
baseline failures and breaks 5 baseline successes, for a net gain of 6 episodes.
Most delayed failures are `click->click`, meaning the action type is correct but
the location or target is wrong. Among newly fixed successes, the dominant error
families are `click->click` and `type->click`.

The mechanism that worked is not generic training for longer. It is a useful
early calibration window in the WHAT/WHERE route.

## Experiment Ledger And Reflections

This section records the major V23 branches in order. The point is to preserve
not just the scores, but also the reason each branch changed the next hypothesis.

### 1. Correct Evaluation Baseline

The correct full SFT model is:

```text
checkpoints/gui360-fullparam-sft-step250
```

Balanced full 1000 baseline:

```text
outputs/gui360_fullparam_sft_step250_balanced_8gpu64_stop_bounded/
  eval_summary_20260624_153846.json

TSR      21.7%
StepSR   68.22%
Progress 34.77%
```

Reflection:

- The first low-TSR result came from evaluating the wrong model. After switching
  to `gui360-fullparam-sft-step250`, the expected baseline was recovered.
- Full eval should use lossless vLLM correction serving, not the direct PyTorch
  cooperative server, because the direct server stalls under high concurrency.

### 2. Train-Split Baseline And Hard-State Mining

Train split full SFT eval:

```text
outputs/v23_visual_transition/train_eval_full_sft_8gpu64_stop/
  eval_summary_20260625_043340.json

TSR      38.9%
StepSR   83.5%
```

Hard-state artifacts:

```text
outputs/v23_visual_transition/train_full_sft_hard_states/
  hard_states.jsonl          # 1642 rows
  hard_state_summary.json

outputs/v23_visual_transition/train_full_sft_hard_replay/
  hard_action_replay.jsonl
```

Reflection:

- Training data should come from train-split hard states, not balanced test hard
  states.
- First-error states are the right unit of offline improvement because the eval
  itself stops at first error.

### 3. K-Sample Matcher Candidate Mining

Candidate artifact:

```text
outputs/v23_visual_transition/train_full_sft_k8_candidates/
  matcher_candidates.jsonl
  summary.json
```

Result:

```text
rows:            1642 / 1642
errors:          0
any_success:     76.43%
mean_best_reward 0.7882
```

Reflection:

- The candidate set proves that many hard states have at least one good action
  reachable from the current model distribution.
- This makes the data valuable for selection, distillation, and repair.
- It does not imply that negative-advantage actor updates are safe.

### 4. Offline Candidate GRPO

Trainer:

```text
v23_visual_transition/train_offline_grpo.py
```

Key result:

```text
final checkpoint: 0.1% TSR / 0.6% StepSR on balanced full eval
earlier step-25: 17.5% TSR / 65.8% StepSR on first-200 slice
```

Reflection:

- Negative-advantage candidate updates damaged output format and collapsed the
  actor.
- Matcher reward is useful for selecting positives or identifying failures, but
  using low-reward samples as broad actor negatives is unsafe.

### 5. Conservative Distillation

Objective:

```text
GT SFT anchor + small weight on best matcher-success candidate
no negative-advantage updates
```

Observed 200-slice result:

```text
18.5% TSR / 66.7% StepSR
```

Reflection:

- Removing negative pressure avoids collapse.
- It was still not stronger than the SFT baseline/current best direction.
- This pushed the work toward structural WHAT/WHERE decomposition rather than
  generic candidate distillation.

### 6. WHAT/WHERE Dataset

Artifacts:

```text
outputs/v23_visual_transition/where_what_train_dataset/
  what_examples.jsonl               # 13,829 rows
  where_examples.jsonl              # 9,271 rows
  paired_where_what_examples.jsonl  # 13,829 rows
  summary.json
```

Validation:

```text
malformed split targets: 0
what examples with location tokens: 0
where examples with non-location keys: 0
```

Reflection:

- GUI action generation naturally factors into WHAT and WHERE.
- This factorization is better aligned with the cooperative LoRA architecture
  than treating all target tokens as one undifferentiated action string.

### 7. WHAT/WHERE Routed SFT

Trainer:

```text
v23_visual_transition/train_where_what_routed_sft.py
```

Mechanism:

```text
WHAT tokens:  function, status, non-location args -> route to expert 1
WHERE tokens: coordinate/start/end spans          -> route to expert 2
```

Full balanced result for step-5:

```text
outputs/v23_visual_transition/eval_where_what_routed_step5_balanced_vllm_1000_stop/
  eval_summary_20260625_095715.json

TSR      22.1%
StepSR   68.49%
Progress 35.02%
```

Reflection:

- This was the first full balanced result above the SFT baseline.
- The useful effect is a structural inductive bias, not simply more SFT steps.

### 8. Route-Only Continuation

Best checkpoint:

```text
checkpoints/v23_where_what_route_only_from_step5_short_fullsave/epoch-0_step-10
```

Full balanced result:

```text
TSR      22.3%
StepSR   69.07%
Progress 35.21%
```

First-200 slice:

```text
TSR      18.5%
StepSR   67.72%
Progress 32.48%
```

Reflection:

- Freezing the actor and calibrating the router gave the current best.
- The improvement is small and fragile. On full 1000 it is roughly net +6
  successes relative to the baseline.
- Continuing route-only from this checkpoint with lower route loss dropped to
  18.0% TSR / 66.94% StepSR on the first-200 slice.

### 9. WHERE-Heavy Continuation

Setup:

```text
start: route-only step10
trainable: route + lora_A_2
```

First-200 result:

```text
18.0% TSR / 66.67% StepSR / 31.73% progress
```

Reflection:

- Opening WHERE expert capacity after the calibrated route checkpoint hurts.
- This suggests the useful part is not "more visual expert training" in a broad
  sense.

### 10. Route-Pairwise Preference

Trainer:

```text
v23_visual_transition/train_route_pairwise.py
```

Setup:

```text
start: routed step5
trainable: route only
positive: GT action
negative: valid matcher-failing hard candidates
families: click->click, type->click, click->type
```

First-200 results:

```text
step5:  18.0% TSR / 67.20% StepSR / 31.92% progress
step10: 18.5% TSR / 66.73% StepSR / 31.65% progress
```

Reflection:

- The objective exposed that hard negatives can have higher model logprob than
  GT actions.
- But applying this pressure globally damaged StepSR/progress.
- Pairwise repair needs to be targeted to actual current-best failures, not all
  candidate groups.

### 11. Communication Gate Diagnostics And Training

Diagnostic script:

```text
v23_visual_transition/analyze_comm_gate_information.py
```

Current-best gate diagnostic:

```text
g12/g21 means near 0.501
WHERE (g21 - g12) around 0.00004
```

Direct gate shaping:

```text
WHERE (g21 - g12) rises to about 0.0203
first-200 step10: 18.5% TSR / 67.00% StepSR / 32.06% progress
```

Behavior-anchored gate shaping:

```text
first-200 step10: 18.5% TSR / 67.20% StepSR / 32.25% progress
```

`comm_21_only` with gate + message matrix:

```text
first-200 step10: 18.5% TSR / 66.87% StepSR / 31.96% progress
```

Ultraweak gate shaping:

```text
first-200 step10: 18.0% TSR / 67.20% StepSR / 31.95% progress
```

Reflection:

- Current best does not meaningfully use comm gates as role-aware channels.
- The gates can be opened mechanically.
- Opening them globally hurts or fails to beat current best.
- Therefore communication should only be revisited with targeted hard-state
  repair supervision, not global WHERE-token role labels.

### 12. Cross-Experiment Pattern

The same pattern appears in every negative branch:

```text
global proxy improves the intended proxy;
some current-best correct states regress;
the small current-best gain disappears.
```

This is exactly why OPD self-distillation should be the next frame. It makes
behavior preservation a first-class term instead of an afterthought.

## What Failed

The following branches should not be full-evaluated without a new reason.

| Branch | First-200 result | Conclusion |
|---|---:|---|
| `where_heavy` from route-only step10 | 18.0% TSR / 66.67% StepSR / 31.73% progress | Opening WHERE expert A after current best hurts. |
| Route-only continuation from current best, lower route loss | 18.0% TSR / 66.94% StepSR / 31.71% progress | Route-only is saturated after first 10 continuation steps. |
| Route-pairwise from routed step5, step10 | 18.5% TSR / 66.73% StepSR / 31.65% progress | Pairwise reward proxy hurts StepSR/progress without preserving behavior. |
| Direct comm-gate shaping, step10 | 18.5% TSR / 67.00% StepSR / 32.06% progress | Opens communication gate but worsens behavior. |
| Behavior-anchored comm-gate, step10 | 18.5% TSR / 67.20% StepSR / 32.25% progress | Better than direct gate shaping, still below current best. |
| `comm_21_only` with anchor, step10 | 18.5% TSR / 66.87% StepSR / 31.96% progress | Moving gate and W21 together does not solve it. |
| Ultraweak comm-gate anchor, step10 | 18.0% TSR / 67.20% StepSR / 31.95% progress | Smaller gate movement still does not recover current best. |

The common failure mode is global proxy optimization. Token-role losses,
candidate pairwise preferences, and global comm-gate shaping all change states
that the current best already handles well. Since the margin over baseline is
small, a few regressions erase the gain.

## Communication Gate Lesson

Comm-gate diagnostics show that the current best does not really use the V13
communication gates as informative WHAT/WHERE channels:

```text
current-best 128-row diagnostic:
  g12/g21 means are near 0.501
  WHERE (g21 - g12) is about 0.00004
```

Direct `comm_gate_only` training can mechanically open the intended WHAT-to-WHERE
channel:

```text
after comm-gate training:
  WHERE (g21 - g12) rises to about 0.0203
```

But all comm-gate variants underperform the route-only current best on the
first-200 eval. Therefore the lesson is not "communication is useless". The
lesson is that global role supervision for communication is the wrong objective.

If communication is revisited, it should be tied to specific first-error repair
states, not all coordinate tokens.

## SVD Alignment

The SVD line should be aligned with V23, but not interpreted as evidence for a
gradient-conflict multi-agent decomposition.

Relevant SVD-derived checkpoints:

```text
checkpoints/gui360-fullparam-sft-step250-svd-standard-r128
checkpoints/gui360-fullparam-sft-step250-svd-cooperative-r128
```

Historical balanced-1000 results:

```text
SVD standard LoRA r128, all modules: 17.9% TSR / 65.0% StepSR / 30.5% progress
PEFT balanced cooperative r128 SVD:   18.6% TSR / 65.3% StepSR / 31.0% progress
V15 RL from SVD step-25:              20.8% TSR / 67.9% StepSR / 34.5% progress
```

The key lesson is that SVD is a good low-rank compression/warm start of the SFT
policy. It is not automatically a principled WHAT/WHERE or ELEMENT/VERB agent
split. SVD decomposes the SFT weight delta into low-rank directions; it does not
prove that those directions correspond to conflicting sub-abilities.

### Relation To Phase 0 Gates

The Phase 0 gates tested the actual conflict claims on GUI-360:

```text
WHAT/WHERE gate:       ALIGNED, global cosine 0.18695
ELEMENT/VERB gate:     NO CONFLICT, global cosine -0.01472 with unstable sign
q/k ELEMENT/VERB gate: NO CONFLICT, global cosine -0.01503
```

Therefore SVD should not be used to override the gate. If gradients do not show a
stable negative conflict, factored-specialization multi-agent remains
unjustified, even if SVD compression is useful.

### SVD Cosine-Report Alignment

The direct alignment test is to run the SVD-derived merged model through the
same cosine-report protocol as the full-SFT model. Do not compare raw SVD
singular vectors to WHAT/WHERE labels directly: SVD factors have rotation/sign
ambiguity, and the meaningful object is the gradient geometry induced by the
merged policy.

Merged SVD-static model used for the comparable gate:

```text
checkpoints/gui360-fullparam-sft-step250-svd-standard-r128-visual-merger-merged
```

Reports:

```text
outputs/multiagent_grounding/phase0_conflict_svd_static_hidden/cosine_report.md
outputs/multiagent_grounding/phase0_conflict_svd_static_projection_qk/cosine_report.md
```

Comparison with the original full-SFT Phase 0 gate:

```text
full-SFT hidden WHAT/WHERE:  ALIGNED, global cosine 0.18695, batch mean 0.15434
SVD-static hidden:          ALIGNED, global cosine 0.02056, batch mean 0.13639

full-SFT q/k WHAT/WHERE:    ALIGNED, global cosine 0.16976
  q_proj 0.15885, k_proj 0.17558

SVD-static q/k:             ALIGNED, global cosine 0.02513
  q_proj -0.01584, k_proj 0.04432, most negative L05.q_proj -0.32523
```

This means the SVD-static model can be aligned with the cosine-report method in
the measurement sense: it preserves the Phase 0 verdict. But it does not create
a factored-specialization basis. SVD makes the global WHAT/WHERE geometry much
weaker and introduces a local negative q-projection locus, yet the hidden gate,
global q/k gate, and k-projection gate remain non-conflicting. The right use is
therefore a geometry-preservation / candidate-source diagnostic, not a license
to split WHAT and WHERE agents.

### SVD As Candidate Source

SVD is still relevant under the second multi-agent basis: candidate-source
orthogonality. Existing eval artifacts show this source has non-identical errors.

Comparison against current best on balanced 1000:

```text
current best: 22.3% TSR / 69.07% StepSR / 35.21% progress
SVD static:   17.4% TSR / 64.61% StepSR / 30.12% progress
SVD coop:      0.0% TSR /  0.70% StepSR /  0.10% progress  # broken routing mismatch
```

Episode-level overlap between current best and SVD static:

```text
current-best successes: 223
SVD-static successes:   174
both succeed:           157
current-only successes: 66
SVD-only successes:     17
both fail:              760
failure Jaccard:        0.902
oracle-union TSR:       24.0%
```

The 17 SVD-only successes are small but real. Their current-best first-error
families are:

```text
click->type:  6
click->click: 4
type->type:   3
type->click:  3
type->swipe:  1
```

This means SVD static is too weak to replace current best, but it may provide
useful alternate candidates for a verifier-select system. The right question is
not "should SVD become an agent?" The right question is:

```text
Does SVD static produce correct candidates on a meaningful subset of current-best failures,
and can a verifier select them without breaking current-best successes?
```

That is a candidate-orthogonality gate, not a gradient-conflict gate.

### Practical Alignment

Use SVD in three limited ways:

1. **Compression baseline**: SVD standard LoRA measures how much full-SFT behavior
  is preserved in low rank.
2. **Warm start**: SVD initialization can help parameter-efficient training, as
  V15 RL from SVD showed.
3. **Candidate source**: SVD static can be one source in a verifier-select
  candidate pool because it has some SVD-only successes.

Do not use the broken SVD cooperative checkpoint as a source; it has 0% TSR due
to routing mismatch.

Do not use SVD to justify factored-specialization agents after the Phase 0 gates
reject the conflict basis.

## OPD Self-Distillation Framing

The next useful objective should separate two sets of states:

```text
Preserve set P:
  states where current best is correct or high-confidence

Repair set R:
  states where current best is wrong and a positive alternative is available
  from GT or matcher-success candidates
```

The student starts from the current best checkpoint. The teacher is also the
current best checkpoint, frozen.

The objective should preserve teacher behavior on P and apply repair gradients
only on R:

```text
L = L_preserve + L_repair + L_param_anchor
```

Where:

```text
L_preserve = KL(student(. | s), teacher(. | s))
             or token-logprob MSE on teacher/GT targets

L_repair   = -log sigmoid(logp_student(a_pos | s) - logp_student(a_neg | s) - margin)

L_param_anchor = small L2 to initial trainable parameters
```

The crucial distinction from previous pairwise training is that pairwise repair
is not applied globally. It is only applied where the current best fails.

## Data Construction For OPD

Do not train on balanced test hard states. Use the train split for training and
reserve the balanced test split for evaluation.

Required training data:

1. Run the current best model on `datasets/gui360-balanced/gui360_train_from_parquet.jsonl`.
2. Mine first-error states from that train evaluation.
3. For each first-error state, store:
   - episode id and step index
   - GT screenshot path
   - goal and GT action history
   - teacher predicted action text and parsed action
   - GT action text
   - matcher reward for teacher action
   - optional K sampled candidates and matcher rewards
   - error family, especially `click->click`, `type->click`, `click->type`
4. Build preserve states from train states where teacher succeeds.

The resulting OPD packet should make the role of each row explicit:

```json
{
  "kind": "repair" | "preserve",
  "episode_id": "...",
  "step_idx": 3,
  "goal": "...",
  "image": "...",
  "history": ["Step 1: ..."],
  "teacher_text": "<tool_call>...</tool_call>",
  "positive_text": "<tool_call>...</tool_call>",
  "negative_text": "<tool_call>...</tool_call>",
  "family": "click_grounding",
  "source": "current_best_first_error"
}
```

## Candidate Positives And Negatives

Repair positives:

- GT action text is always a conservative positive.
- Matcher-success sampled candidates can be positives if reward >= 0.5.
- Prefer positives that match the target state transition and preserve format.

Repair negatives:

- Teacher's own failed action is the most important negative.
- Other valid matcher-failing candidates are secondary negatives.
- Do not use malformed outputs as primary negatives unless the failure being
  repaired is format collapse.

This is the opposite of broad GRPO: the actor is not pushed away from every low
reward sample. It is pushed away from the actual current-best mistake on a state
where a reliable positive exists.

## First OPD Experiment

Start with a conservative student:

```text
base checkpoint: current best route-only step10
trainable params: route_only
teacher: same current best, frozen
repair data: current-best train first-error states only
preserve data: current-best train correct states
max steps: 5 to 10
LR: 1e-4 to 2e-4 for route weights
```

Loss sketch:

```text
repair rows:
  L = pairwise_repair + alpha * teacher_anchor_on_positive

preserve rows:
  L = beta * teacher_anchor

all rows:
  L += gamma * route_l2_to_initial
```

A reasonable first setting:

```text
repair_weight = 1.0
preserve_weight = 1.0
teacher_anchor_weight = 5.0 to 20.0
route_l2_weight = 1.0
margin = 0.0
```

If route-only OPD is stable but not helpful, a second run can try:

```text
trainable params: route + comm_gate_21
```

Do not start by training LoRA A/B or comm W. Previous runs show that opening
larger actor capacity easily erases current-best behavior.

## Evaluation Gate

No new branch should be full-evaluated unless it beats current best on the same
first-200 slice:

```text
current best first-200:
  TSR      18.5%
  StepSR   67.72%
  Progress 32.48%
```

The preferred acceptance condition is:

```text
StepSR > 67.72%
Progress > 32.48%
TSR >= 18.5%
```

If a branch only ties TSR but lowers StepSR or progress, do not run full 1000.

## Why This Is The Right Next Step

All V23 negative results point to the same constraint: the current best is a
small but fragile improvement. Global objectives repair some states and break
others. OPD self-distillation turns this into an explicit optimization problem:

```text
keep the teacher distribution where it works;
change it only where the teacher demonstrably fails;
measure every change on first-error survival.
```

This is closer to the actual evaluation geometry than global route labels,
global comm-gate role labels, or unconstrained candidate GRPO.