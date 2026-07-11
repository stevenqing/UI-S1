# Not Every Correction Teaches

## Source-Conditioned Revision Utility for Heterogeneous GUI Agents

## Research question

Can heterogeneous GUI agents produce diverse failed trajectories that another agent can globally revise into useful training supervision?

The completed experiment separates four questions that are often conflated:

1. Are the actor failures diverse?
2. Is a revision better than its source actor action?
3. Does a revised action sequence form a coherent trajectory?
4. Does training a stronger student on those revisions improve held-out behavior?

The answer is not uniformly positive. Heterogeneous actors produce highly diverse errors, and the global corrector has positive average utility relative to weak source actors, but unfiltered full-parameter training substantially harms the held-out student.

## Protocol

### Actors and corrector

- Actors: Qwen3-VL-8B and InternVL3-8B.
- Global corrector: Qwen3.5-9B.
- The actor and corrector are always different models.
- Actors operate on teacher-forced GUI-360 screenshots and action histories.
- The corrector sees the complete trajectory and screenshot sequence, including future screenshots.
- Frozen matcher scores are diagnostic only and are not used to select the formal noisy training set.

### Scale

- Train: 1,573 GUI-360 episodes / 12,574 steps.
- Actor trajectories: 3,146.
- Error trajectories: 3,076.
- Structurally usable corrected trajectories: 2,128.
- Unique noisy SFT rows: 14,456.
- Held-out test: 1,000 episodes / 7,498 steps.

### Formal training

- Starting checkpoint: GUI-360 full-parameter SFT step 250.
- Six GPUs with DeepSpeed ZeRO-3.
- 8.292B / 8.292B trainable parameters.
- Vision tower, multimodal projector, and language model unfrozen.
- Effective batch 48, learning rate 6e-6, one epoch, 302 optimizer steps.

## Finding 1: diversity is not utility

The heterogeneous actors have:

- 91.68% action disagreement.
- 71.11% error-set Jaccard.

This establishes substantial failure diversity, but diversity alone does not imply useful supervision.

## Counterfactual Revision Utility

For frozen matcher reward $M$, define source-relative step utility:

$$
 u_t^{src}=M(a_t^{rev})-M(a_t^{actor})\in\{-1,0,+1\}.
$$

This induces four outcomes:

| Outcome | Actor | Revision | Utility |
|---|---|---|---:|
| rescue | wrong | correct | +1 |
| regress | correct | wrong | -1 |
| preserve-correct | correct | correct | 0 |
| unresolved | wrong | wrong | 0 |

Across all 14,456 usable steps:

| Outcome | Count | Fraction |
|---|---:|---:|
| rescue | 1,480 | 10.24% |
| regress | 827 | 5.72% |
| preserve-correct | 2,284 | 15.80% |
| unresolved | 9,865 | 68.24% |

Actor accuracy on this subset is 21.52%; revised accuracy is 26.04%. Net actor-relative utility is +4.52pp, with a trajectory-clustered 10,000-draw bootstrap interval of [+3.61pp, +5.43pp]. However, only 44 of 2,128 complete trajectories are fully rescued (2.07%).

## Finding 2: revision utility is source-conditioned

The same Qwen3.5 corrector has opposite effects by actor source:

| Source | Actor acc | Revised acc | Net utility | Rescue / regress | Trajectory rescue |
|---|---:|---:|---:|---:|---:|
| InternVL3 | 11.57% | 27.62% | +16.05pp | 1,075 / 210 | 3.48% |
| Qwen3-VL | 27.44% | 25.10% | -2.34pp | 405 / 617 | 0.99% |

Correction quality is therefore a relation between source, corrector, state, and action—not an intrinsic property of the corrector output.

## Finding 3: prefix consistency is a separate requirement

The corrector rewrites actions over a fixed ground-truth screenshot sequence. SFT histories contain the revised action prefix, while screenshots remain teacher-forced. Define a diagnostic clean prefix as one in which every earlier revised action matches the frozen reference.

| Prefix | Rows | Fraction | Current-label accuracy |
|---|---:|---:|---:|
| clean | 3,796 | 26.26% | 45.10% |
| dirty | 10,660 | 73.74% | 19.25% |

Current-label accuracy falls monotonically with previous matcher-wrong revisions:

| Previous wrong revisions | Accuracy |
|---:|---:|
| 0 | 45.10% |
| 1 | 25.41% |
| 2 | 19.41% |
| 3 | 17.90% |
| 4+ | 16.93% |

This is a diagnostic proxy rather than an executed transition-equivalence test. It nevertheless exposes an important constraint: globally rewriting actions does not guarantee that the revised action prefix causally reaches the fixed future screenshot.

## Finding 4: source-relative gain does not imply student utility

The full noisy-SFT run reaches aggregate train loss 0.2131, but held-out performance changes as follows:

| Metric | Baseline | Post | Delta |
|---|---:|---:|---:|
| TSR | 18.70% | 0.90% | -17.80pp |
| Step accuracy | 57.10% | 35.62% | -21.47pp |

The paired 10,000-draw TSR bootstrap interval is [-20.20pp, -15.50pp]. There are zero task wrong-to-right flips and 178 task right-to-wrong flips.

Being better than weak source actors is not sufficient for a revision set to supervise a stronger starting student.

## Confidence caveat

- 88.11% of usable trajectories have self-reported confidence 0.95.
- Confidence vs revised step accuracy Spearman: 0.009.
- Confidence vs net revision utility Spearman: 0.022.
- Confidence AUC for complete-trajectory rescue: 0.638.

Correction confidence is retained in raw metadata but is absent from the ShareGPT conversations used for SFT. It is a weak selection/calibration diagnostic and cannot be claimed as a causal feature learned by the student.

## Revision Utility Ladder

A corrected trajectory should be evaluated at five distinct levels:

1. Structural utility: can the output be parsed and executed?
2. Source-relative utility: is the revision better than its source actor?
3. Prefix/transition consistency: can the revised prefix reach subsequent states?
4. Student-relative utility: is the revision better than the student being trained?
5. Downstream learning utility: does training improve held-out behavior?

Passing a lower level does not imply passing a higher level.

## Proposed method direction

A source- and student-conditioned revision gate should estimate:

$$
 u_t^{src}=M(a_t^{rev})-M(a_t^{actor}),\qquad
 u_t^{stu}=M(a_t^{rev})-M(a_t^{student}).
$$

Candidate gate inputs include source identity, actor/revision/student disagreement, corrector self-consistency, prefix consistency, transition evidence, and verifier scores. Its decision space is:

```text
use_revision | keep_student | keep_actor | reject/replan
```

A conservative preference objective follows directly:

- Rescue: revision preferred to actor.
- Regress: actor preferred to revision.
- Preserve-correct: clean replay or consistency training.
- Unresolved: mask, reject, or replan.

## Required causal controls

The next experiment should cross target label and history source:

| Arm | Target | History | Purpose |
|---|---|---|---|
| A0 | no update | — | frozen baseline |
| A1 | GT | GT | positive control |
| A2 | random/marginal matched | GT | negative control |
| A3 | actor | actor | actor imitation |
| A4 | revision | revision | completed treatment |
| A5 | revision | GT | isolate target noise |
| A6 | GT | revision | isolate context mismatch |
| A7 | revision | clean prefix only | test sequential contamination |
| A8 | revision | dirty prefix only | harm control |

The starting student must also be evaluated on the same 14,456 states to measure student-relative rescue and regression. Initial screening should use LoRA; only methods that preserve held-out step accuracy should advance to full-parameter confirmation.

## Positioning

Trajectory refinement itself is not new. Relevant work includes AgentRefine, Agent-R, STeCa, STeP, GUI-Reflection, UI-Genie, UI-Voyager, STEVE, V-Droid, GAIA, and weak-to-strong trust filtering. The differentiating research object here is the conjunction of:

- heterogeneous GUI source actors,
- source- and student-conditioned revision utility,
- step-to-trajectory composition,
- teacher-forced prefix consistency,
- and downstream paired causal evaluation.

## Reproducibility

Core implementation:

- [Trajectory generation and global revision](../scripts/multiagent_trajectory_revision.py)
- [Counterfactual revision utility analysis](../scripts/analyze_multiagent_revision_utility.py)
- [Full-parameter data and config preparation](../scripts/prepare_multiagent_fullparam_llamafactory.py)
- [Held-out evaluator](../scripts/evaluate_multiagent_revision_pilot.py)
- [Exact shard merger](../scripts/merge_multiagent_revision_eval.py)
- [Paired report generator](../scripts/report_multiagent_revision_training.py)

Raw trajectories, model checkpoints, screenshots, and generated outputs are intentionally excluded by the repository ignore policy.
