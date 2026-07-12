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

Within actor × absolute-step strata containing both clean and dirty examples, overlap weighting still gives a **+22.35pp** clean-minus-dirty accuracy difference over 1,588 overlap-weighted rows. This reduces a simple source/position explanation, but remains descriptive rather than causal.

This is a diagnostic proxy rather than an executed transition-equivalence test. It nevertheless exposes an important constraint: globally rewriting actions does not guarantee that the revised action prefix causally reaches the fixed future screenshot.

## Finding 4: source-relative gain does not imply student utility

The frozen starting checkpoint was evaluated greedily on the exact 14,456 A4 state/history rows used by the formal noisy treatment. Relative to that student:

| Outcome | Count | Fraction |
|---|---:|---:|
| Revision rescues student | 881 | 6.09% |
| Revision regresses student | 4,981 | 34.46% |
| Both correct | 2,883 | 19.94% |
| Both wrong | 5,711 | 39.51% |

- Starting-student accuracy: **54.40%**.
- Revision accuracy: **26.04%**.
- Net student-relative revision utility: **-28.36pp**.
- Trajectory-clustered 95% interval: **[-29.58pp, -27.18pp]**.
- Given a student-correct step, the revision regresses it **63.34%** of the time.

The negative student-relative utility occurs for both sources: -29.18pp for InternVL3-derived revisions and -27.88pp for Qwen3-VL-derived revisions. This directly resolves the earlier apparent paradox: the corrector can improve a very weak source actor while still being dramatically worse than the student that receives the supervision.

The full noisy-SFT run reaches aggregate train loss 0.2131, but held-out performance changes as follows:

| Metric | Baseline | Post | Delta |
|---|---:|---:|---:|
| TSR | 18.70% | 0.90% | -17.80pp |
| Step accuracy | 57.10% | 35.62% | -21.47pp |

The paired 10,000-draw TSR bootstrap interval is [-20.20pp, -15.50pp]. There are zero task wrong-to-right flips and 178 task right-to-wrong flips.

Being better than weak source actors is not sufficient for a revision set to supervise a stronger starting student; the same-state measurement now quantifies that gap directly.

## Finding 5: history mismatch is real but context repair cannot rescue wrong labels

On a paired 2,048-row grid balanced over actor source × diagnostic prefix cleanliness, replacing revised history with GT history changes the frozen starting student's behavior on 39.75% of steps:

- Revision-history student accuracy: 62.11%.
- GT-history student accuracy: 73.29%.
- Balanced-grid delta: **+11.18pp**.
- Population-standardized delta over the full 14,456-row actor×prefix distribution: **+16.47pp**.
- Wrong→right flips: 281; right→wrong flips: 52.

The effect is concentrated in dirty prefixes: +22.07pp for InternVL3-derived rows and +22.46pp for Qwen3-VL-derived rows. Clean-prefix effects are near zero. This supports the history–screen inconsistency mechanism.

However, the LoRA target × history factorial reveals an interaction rather than two additive harms:

| Effect on held-out step accuracy | Estimate | Episode-bootstrap 95% interval |
|---|---:|---:|
| GT history effect given GT target (A1−A6) | +6.67pp | [+4.18pp, +9.15pp] |
| GT history effect given revision target (A5−A4) | **-11.04pp** | [-14.29pp, -7.75pp] |
| Revision-label effect under GT history (A5−A1) | **-30.05pp** | [-35.72pp, -24.40pp] |
| Revision-label effect under revision history (A4−A6) | -12.35pp | [-17.62pp, -7.16pp] |
| Label × history interaction | **-17.70pp** | [-21.85pp, -13.37pp] |

Correct history helps when targets are correct, but makes wrong revision targets more directly learnable. Noisy revised history partially masks wrong-label supervision rather than repairing it. Therefore, fixing context alone is insufficient; target utility must be gated first.

## Finding 6: equal-budget LoRA controls validate the causal pipeline

All arms use 800 rows, 100 optimizer updates, identical LoRA configuration, and the same frozen 125-episode / 915-step held-out screen:

| Arm | Role | ΔTSR | Δstep | Gate |
|---|---|---:|---:|---|
| A1 GT target + GT history | positive control | +1.60pp | +0.44pp | HELPS |
| A6 GT target + revision history | context control | +0.80pp | -6.23pp | HARMS |
| A10 Qwen3-VL-source revision | source gate | -16.00pp | -15.30pp | HARMS |
| A9 InternVL3-source revision | source gate | -16.00pp | -17.27pp | HARMS |
| A4 revision + revision history | treatment | -17.60pp | -18.58pp | HARMS |
| A5 revision + GT history | history intervention | -18.40pp | -29.62pp | HARMS |
| A7 clean-prefix revision | oracle prefix control | -22.40pp | -21.75pp | HARMS |
| A2 marginal-matched random target | negative control | -23.20pp | -51.04pp | HARMS |

The positive control shows that the trainer and update budget can produce a non-harmful improvement; the random control establishes the expected lower bound. Source-only and prefix-only selection fail because every such subset remains strongly negative relative to the starting student. No deployable candidate advanced to full-grid or full-parameter confirmation.

## Finding 7: metadata-only utility prediction is insufficient

An episode-disjoint logistic gate used source identity, action types, candidate agreement, position, confidence, and non-oracle prefix metadata while excluding GT actions and matcher outcomes. On 1,666 held-out rows:

- Rescue base rate: 4.98%.
- ROC-AUC: 0.6101.
- Average precision: 0.0678.
- Every nonzero-coverage operating point had negative accepted-set net utility.

Simple metadata cannot identify the rare revision rescues. The next selector must semantically inspect the screenshot, goal, history, revision rationale, and actor/revision/student candidate packet.

An episode-disjoint multimodal verifier dataset has therefore been constructed with three conservative decisions:

- `keep_student`: 6,116 train rows before balancing.
- `use_revision`: 717 train rows before balancing.
- `replan`: 4,606 train rows before balancing.
- Balanced training set: 2,048 examples per decision, 6,144 total.

This verifier-agent formulation is now the main positive method direction.

## Finding 8: perfect rescue selection still needs replay or conservative optimization

An oracle ceiling selected 800 revision rows where the starting student was wrong and the revision was matcher-correct, with GT histories. Despite 100% target correctness, direct 100-update LoRA SFT still harmed the screen:

| Rescue / clean replay | ΔTSR | Δstep | Gate |
|---|---:|---:|---|
| 100% / 0% (A13) | -10.40pp | -12.57pp | HARMS |
| 50% / 50% (A14) | -4.80pp | -4.48pp | HARMS |
| 25% / 75% (A15, 125-episode screen) | +1.60pp | -1.75pp | NO CLEAR SIGNAL |
| 10% / 90% (A16) | -2.40pp | -2.95pp | HARMS |

Clean replay strongly reduces forgetting, but the small screen is non-monotonic. On the complete 1,000-episode / 7,498-step grid, A15 provides the first statistically positive result:

- TSR: 18.70% → **21.50%** (**+2.80pp**).
- Paired 10,000-draw TSR interval: **[+1.00pp, +4.60pp]**.
- Step accuracy: 57.10% → **57.27%** (+0.17pp).
- Task wrong→right / right→wrong: 56 / 28.
- Step wrong→right / right→wrong: 552 / 539.

This validates the central positive hypothesis: a student-relative rescue subset can improve held-out task success when coupled with enough clean replay. A15 remains an oracle ceiling because selection uses matcher/student correctness, but it proves that the heterogeneous revision bank contains real learning signal rather than only noise.

The result also establishes that utility gating and label correctness alone are insufficient: the accepted-set distribution and preservation objective matter. The actionable target is now to approximate A15's oracle selection without GT while maintaining roughly a 1:3 rescue-to-clean replay ratio.

The preferred training direction is therefore calibrated rescue ranking followed by 25% accepted revisions + 75% clean replay, or conservative preference/KL-regularized updates—not SFT on accepted rescue rows alone.

## Finding 9: verifier class prior exposes a precision–recall failure

The balanced multimodal verifier fits its training data but is unsafe on the episode-disjoint original-distribution test:

- Decision accuracy 58.58%, macro-F1 0.4572.
- `use_revision` precision 10.95%, recall 18.07%.
- Unsafe overwrite rate 3.18%.
- Fallback-student routed accuracy 54.92% versus 57.20% baseline.

A dev-only search over non-oracle source/type/agreement rules found no rule with at least ten accepted dev examples and positive rescue-minus-regression utility; the fail-closed locked test policy therefore has zero coverage.

An equal-update natural-prior verifier eliminates unsafe overwrites but collapses `use_revision` recall to zero:

- Decision accuracy 65.61%, macro-F1 0.4448.
- `use_revision` precision/recall 0/0.
- Fallback-student accuracy exactly preserves the 57.20% baseline.

Thus balanced training over-commits rare revisions, while natural training suppresses them entirely. The next method requires calibrated selective classification/ranking of `use_revision`, not plain balanced three-way SFT.

## Finding 10: calibrated binary rankers still cannot identify positive-utility revisions

Three episode-disjoint multimodal binary rankers were trained to predict whether a revision rescues the starting student. All use continuous conditional $P(\mathrm{YES})/(P(\mathrm{YES})+P(\mathrm{NO}))$ scores and select thresholds using dev rescue-minus-regression utility only.

| Ranker | Train negatives | Repeat policy | Dev AUC / AP | Test AUC / AP | Locked utility gate |
|---|---|---|---:|---:|---|
| v1 | regress + both-correct + both-wrong | rescue oversampled | 0.617 / 0.085 | 0.647 / 0.086 | no positive dev threshold |
| v2 | regress only | rescue oversampled | 0.538 / 0.070 | 0.589 / 0.078 | dev +0.15pp, test **-0.42pp** |
| v3 | regress only | no repeated examples | 0.466 / 0.054 | 0.523 / 0.056 | no positive dev threshold |

The v2 dev threshold accepts 89 rows with 10 rescues and 8 regressions, but the locked test rule accepts 135 rows with 13 rescues and 20 regressions. Removing repeated rescue examples further reduces ranking quality. Binary re-framing, cost-sensitive negatives, and continuous thresholding therefore do not solve selector generalization with the current backbone.

## Finding 11: simple visual-transition consistency also fails to generalize

For offline data curation, a GT-action-free gate compared the local $s_t\rightarrow s_{t+1}$ pixel-change mass around revision and student click coordinates. The threshold was selected on dev using only future screenshots and candidate coordinates:

- Dev: 54 accepted, 14 rescues, 8 regressions, population utility **+0.44pp**.
- Locked test: 56 accepted, 14 rescues, 15 regressions, population utility **-0.06pp**.

The next screenshot contains useful local transition evidence, but simple pixel-change localization is not stable enough to approximate student-relative utility.

## Finding 12: Pass@8 fixed-choice selection crosses the non-oracle utility gate

A new selector-fresh, episode-disjoint split was frozen over the 962 cached critical steps before selector inference: 23 smoke, 231 dev, and 708 locked-test steps. Blind packets retain exact actions from four K=8 proposal sources but remove GT, rewards, correctness, source identity, and GT-derived diagnostics. Both dev and locked predictions were completed before either label split was opened.

| Selector | Dev utility | Locked utility | Locked 95% episode-cluster CI | Rescue / regress | Oracle capture |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B fixed-choice | +6.49pp | **+6.36pp** | **[+4.53,+8.31]pp** | 46 / 1 | **20.93%** |
| Qwen3.5-35B-A3B fixed-choice | +2.60pp | +4.52pp | [+2.93,+6.20]pp | 33 / 1 | 14.88% |
| Exact plurality | +3.03pp | +3.25pp | [+1.87,+4.73]pp | 25 / 2 | 10.70% |
| Cross-source consensus | +5.63pp | +5.37pp | [+3.61,+7.25]pp | 39 / 1 | 17.67% |

The locked packet oracle is 30.79% versus a 0.42% frozen-student baseline. Qwen3.5-9B reaches 6.78% selected accuracy. Scaling the selector does not help: 35B-minus-9B is -1.84pp with 95% CI [-3.69,+0.00]pp. The 35B model changes only 34.18% of actions versus 60.03% for 9B. Cross-source consensus is close to 9B (-0.99pp, CI [-2.98,+1.12]pp), showing that repeated independent proposal support explains much of the usable signal.

## Updated selector boundary

The experiment now has both a positive learning ceiling and a positive non-oracle **hard-step candidate-recovery** result:

- **Positive training ceiling:** oracle student-rescue selection + 75% clean replay gives full-grid TSR +2.80pp with preserved step accuracy.
- **Positive proposal selector:** frozen Qwen3.5-9B selection converts Pass@8 diversity into +6.36pp critical-step utility with a positive locked interval.
- **Scale hypothesis rejected:** Qwen3.5-35B-A3B is more conservative and does not outperform 9B.
- **Consensus is a strong control:** zero-GPU cross-source consensus reaches +5.37pp and is not significantly worse than 9B.

This does not establish an arbitrary-state online router: the underlying benchmark episodes are not benchmark-fresh, and the target population was historically selected with GT-conditioned critical diagnostics. The gate authorizes a **selector-to-training bridge study**, not direct SFT. No dev/locked row may enter training, and full-policy held-out evaluation is still required.

## Finding 13: positive utility does not imply training-ready label purity

On student-wrong states, choosing another wrong action has utility zero but remains an actively wrong SFT target. Exact post-hoc diagnosis of frozen locked outputs shows:

| GT-free construction | Changed rows | Correct labels | SFT purity | Wilson 95% |
|---|---:|---:|---:|---:|
| All Qwen3.5-9B changes | 425 | 46 | **10.82%** | [8.21%,14.14%] |
| All cross-source-consensus changes | 334 | 39 | **11.68%** | [8.66%,15.56%] |
| 9B/consensus same-action intersection | 114 | 13 | **11.40%** | [6.79%,18.54%] |

Thus neither consensus nor the 9B/consensus intersection is currently training-ready; the intersection reduces coverage without increasing locked purity. Qwen3.5-9B also enriches actions containing its own exact source by 1.36×, but self-only selections have just 6.99% purity, while Qwen3.5-plus-another-source selections reach 18.52%. This supports independent agreement rather than self-source recognition as the useful mechanism.

Before any selected-revision SFT, two bridge quantities are mandatory: (1) a controlled P100/P80/P60/P40 purity-response curve at fixed 25% revision + 75% clean replay, and (2) aggregate train-split purity for frozen all-9B, consensus, and same-action-intersection constructions. A variant is eligible only if its purity lower confidence bound exceeds the empirically tolerated training-purity threshold. A separate uniformly sampled student-correct control must measure regression risk.

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

The starting-student evaluation is complete. Initial training screening uses LoRA; only deployable methods that preserve held-out step accuracy advance to full 1,000-episode and then full-parameter confirmation.

## Follow-up implementation status

The pre-registered data matrix is now materialized from the exact 14,456-row formal treatment grid. The A4 core fields (`history`, action key, target text, image, and target ID) match the data used by the completed full-parameter run with zero mismatches.

| Arm | Rows | Diagnostic target accuracy |
|---|---:|---:|
| A1 GT target + GT history | 14,456 | 99.88% |
| A2 marginal-matched random target + GT history | 14,456 | 4.43% |
| A3 executable actor target + actor history | 11,722 | 26.54% |
| A4 revision target + revision history | 14,456 | 26.04% |
| A5 revision target + GT history | 14,456 | 26.04% |
| A6 GT target + revision history | 14,456 | 99.88% |
| A7 revision + diagnostic clean prefix | 3,796 | 45.10% |
| A8 revision + diagnostic dirty prefix | 10,660 | 19.25% |
| A9 InternVL3-source revisions | 5,391 | 27.62% |
| A10 Qwen3-VL-source revisions | 9,065 | 25.10% |
| A11 oracle matcher-correct revisions | 3,764 | 100.00% |
| A12 oracle matcher-correct + clean prefix | 1,712 | 100.00% |

A7 is a matcher-defined conditional filtering policy and is an oracle diagnostic control. Even if it improves downstream behavior at the same update budget, that would not by itself prove that prefix cleanliness causally causes the gain, because the accepted subset may be easier in other ways.

A matched greedy evaluator and exact shard merger completed all 14,456 A4 rows. The A5 intervention, equal-budget LoRA screens, oracle replay confirmation, and frozen Pass@8 selector study are complete. Candidate-packet verification now passes on hard-step utility, but selected-label purity remains only 10–12%. The current next stage is therefore the pre-registered purity-response/train-purity bridge; formal 25/75 policy training remains blocked until that bridge passes.

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
- [Causal-arm data builder](../scripts/build_revision_causal_arms.py)
- [Matched starting-student evaluator](../scripts/evaluate_revision_causal_arm.py)
- [Exact causal-eval shard merger](../scripts/merge_revision_causal_eval.py)
- [Student-relative utility analysis](../scripts/analyze_student_relative_revision.py)
- [Target × history factorial analysis](../scripts/analyze_revision_lora_factorial.py)
- [Metadata utility-gate baseline](../scripts/train_revision_utility_gate.py)
- [Student-rescue oracle builder](../scripts/build_student_rescue_oracle_arm.py)
- [Multimodal revision-verifier data builder](../scripts/build_revision_verifier_data.py)
- [Full-parameter data and config preparation](../scripts/prepare_multiagent_fullparam_llamafactory.py)
- [Held-out evaluator](../scripts/evaluate_multiagent_revision_pilot.py)
- [Exact shard merger](../scripts/merge_multiagent_revision_eval.py)
- [Paired report generator](../scripts/report_multiagent_revision_training.py)

Raw trajectories, model checkpoints, screenshots, and generated outputs are intentionally excluded by the repository ignore policy.
