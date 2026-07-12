# Not Every Correction Teaches

## Source-, Student-, and Context-Conditioned Revision Utility for GUI Agents

## One-sentence thesis

Heterogeneous agents provide diverse GUI failures, but a revision becomes useful supervision only when it rescues the target student and is mixed with sufficient clean replay to prevent distributional forgetting.

## Draft abstract

Self-improving GUI agents increasingly rely on synthetic trajectories produced or revised by other agents, yet trajectory diversity and corrector confidence do not establish downstream training value. We conduct a controlled study over 1,573 GUI-360 training episodes and 1,000 held-out episodes using Qwen3-VL and InternVL3 actors, a Qwen3.5 global trajectory corrector, and an 8.29B-parameter GUI student. The actors exhibit 91.68% action disagreement, but unfiltered corrected labels are only 26.04% accurate and full-parameter training reduces held-out task success by 17.80 points. We formalize Counterfactual Revision Utility, separating rescue, regression, preservation, and unresolved outcomes. On the exact 14,456 training states, revisions are 28.36 points worse than the starting student despite being 4.52 points better than their weak source actors. We further identify a teacher-forced prefix-consistency gap: replacing revised histories with ground-truth histories improves frozen-student accuracy by a population-standardized 16.47 points, but a target-by-history factorial shows that clean context makes wrong labels more directly learnable. Training on 25% oracle student-rescue revisions and 75% clean replay improves task success from 18.70% to 21.50% on all 1,000 held-out episodes, with a paired bootstrap interval of [+1.00,+4.60] points. Earlier metadata, verifier, ranker, and transition selectors fail, but a newly frozen Pass@8 fixed-choice study crosses the non-oracle hard-step gate: Qwen3.5-9B obtains +6.36 points rescue-minus-regression utility with interval [+4.53,+8.31]. Scaling the selector to Qwen3.5-35B-A3B does not help, while zero-GPU cross-source consensus reaches +5.37 points. The results identify proposal agreement and student-relative selection—not corrector scale or confidence—as the key bridge from diverse synthetic candidates to useful supervision.

## Core research questions

1. Does heterogeneous actor disagreement imply useful synthetic supervision?
2. Is revision quality intrinsic, or conditioned on the source actor and target student?
3. How does teacher-forced history–screen inconsistency affect revision learning?
4. Can student-rescue selection plus clean replay recover positive downstream utility?
5. Can a non-oracle fixed-choice selector convert Pass@8 proposal diversity into positive student-relative utility, and does selector scale help?

## Contributions supported by completed evidence

### C1. Heterogeneous failure bank and paired causal protocol

- 1,573 train episodes / 12,574 steps.
- 3,146 actor trajectories, 3,076 error trajectories.
- Qwen3-VL and InternVL3 action disagreement: 91.68%.
- Frozen 1,000-episode / 7,498-step held-out grid with exact sharded merge.

### C2. Counterfactual Revision Utility

For matcher $M$:

$$
 u_t^{src}=M(a_t^{rev})-M(a_t^{actor}),\qquad
 u_t^{stu}=M(a_t^{rev})-M(a_t^{student}).
$$

The four outcomes are rescue, regress, preserve-correct, and unresolved.

### C3. Source–student utility reversal

- Source-relative revision utility: +4.52pp, CI [+3.61,+5.43]pp.
- Student-relative revision utility: -28.36pp, CI [-29.58,-27.18]pp.
- Revisions regress 4,981 student-correct steps while rescuing only 881 student-wrong steps.

### C4. Prefix-consistency and target×history interaction

- GT history intervention: +11.18pp balanced / +16.47pp population-standardized.
- With GT targets, GT history improves held-out step accuracy +6.67pp.
- With revision targets, GT history reduces held-out step accuracy -11.04pp.
- Label×history interaction: -17.70pp.

### C5. Statistically positive utility-selection ceiling

A15 uses 200 oracle student-rescue revisions and 600 clean-replay rows, trained for 100 LoRA updates:

- TSR: 18.70% → 21.50%, +2.80pp.
- Paired 10k bootstrap: [+1.00,+4.60]pp.
- Step accuracy: 57.10% → 57.27%, +0.17pp.
- Task wrong→right / right→wrong: 56 / 28.

This is an oracle ceiling, not a deployable method claim.

### C6. Precision–recall failure of naive verification

- Metadata gate: AP 6.78%; every nonzero-coverage threshold has negative net utility.
- Balanced multimodal verifier: use-revision precision 10.95%, recall 18.07%; routing harms by 2.28pp.
- Natural-prior verifier: zero unsafe overwrite but zero use-revision recall.
- Dev-only conservative rules correctly fail closed at zero coverage.

### C7. Failure boundary across calibrated selector families

- Binary v1 reaches test AUC/AP 0.647/0.086 but has no positive-utility dev threshold.
- Cost-sensitive rescue-vs-regress v2 finds a weak positive dev threshold (+0.15pp) that reverses on test (-0.42pp).
- Unique non-oversampled v3 falls to near-random test AUC/AP 0.523/0.056.
- A future-screenshot pixel-transition gate is positive on dev (+0.44pp) but neutral/negative on locked test (-0.06pp).

These failures make the contribution sharper: the revision bank contains useful signal, but identifying student rescue is substantially harder than predicting generic correctness or transition proximity.

### C8. Pass@8 proposal selection is positive, but larger correctors are not better

A selector-fresh episode split over 962 cached hard steps evaluates identical anonymized K=8 candidate packets. Qwen3.5-9B obtains locked rescue-minus-regression utility **+6.36pp**, episode-cluster CI **[+4.53,+8.31]pp**, and captures 20.93% of packet-oracle headroom. Qwen3.5-35B-A3B also passes (+4.52pp) but underperforms 9B by 1.84pp (CI [-3.69,+0.00]pp). A zero-GPU cross-source-consensus rule reaches +5.37pp and is not significantly different from 9B.

This changes the boundary from “no non-oracle selector” to “positive hard-step candidate recovery, not yet a full-policy method.” Pass@8 diversity is usable, but selector scale is not the governing mechanism; independent proposal agreement explains much of the gain.

## Main result table

| Experiment | Scale | ΔTSR | Δstep | Conclusion |
|---|---:|---:|---:|---|
| Unfiltered fullparam revisions | 1,000 episodes | -17.80pp | -21.47pp | Strong harm |
| GT positive control LoRA | 125 episodes | +1.60pp | +0.44pp | Pipeline can help |
| Source-only revisions | 125 episodes | -16.00pp | -15.30 to -17.27pp | Source gating insufficient |
| Clean-prefix revisions | 125 episodes | -22.40pp | -21.75pp | Prefix gating insufficient |
| 100% oracle rescue | 125 episodes | -10.40pp | -12.57pp | Correct labels still forget |
| 50% rescue / 50% replay | 125 episodes | -4.80pp | -4.48pp | Replay mitigates |
| 25% rescue / 75% replay | 1,000 episodes | **+2.80pp** | **+0.17pp** | Positive oracle ceiling |

## Proposed method after diagnosis

### Calibrated Student-Relative Rescue Ranker

Input:

- screenshot, goal, and current history;
- source actor identity and candidate;
- starting-student candidate;
- global revision and rationale;
- non-oracle agreement and consistency evidence.

Output:

$$
 p_\phi(\text{revision rescues student}\mid x,a^{actor},a^{student},a^{rev}).
$$

The threshold is selected on episode-disjoint dev rescue-minus-regression utility, not classification accuracy.

The current GUI student backbone does not satisfy this goal under three binary training variants. However, frozen fixed-choice inference on a newly frozen selector split does cross the hard-step utility gate. The best model is Qwen3.5-9B rather than Qwen3.5-35B-A3B, and cross-source consensus is competitive. The next causal test therefore moves selection to a disjoint train split and evaluates the validated 25/75 arm on a full held-out policy grid.

### Conservative action learning

- Select high-precision revisions only.
- Keep approximately 25% accepted revisions and 75% broad clean replay.
- Prefer pairwise revision-vs-student optimization or KL-regularized SFT.
- Fail closed when no dev threshold has positive utility.

## Required next positive-result bar

A deployable selector succeeds only if:

1. Its threshold is fixed using dev only.
2. Test accepted-set rescue exceeds regression.
3. A 25/75 selected-revision/replay LoRA arm preserves step accuracy within -1pp.
4. Full 1,000-episode TSR improves with a paired interval excluding zero.
5. A second corrector or benchmark reproduces the result.

## Claims that remain unsupported

- Global trajectory correction is universally ineffective.
- Positive critical-step selection necessarily yields positive full-policy TSR after training.
- Qwen3.5-35B-A3B is a stronger GUI action selector than Qwen3.5-9B.
- The hard-step selector is safe as an arbitrary-state online router.
- The 25/75 ratio is universally optimal.
- Teacher-forced prefix inconsistency is the sole cause of negative transfer.
- Findings generalize beyond Qwen3.5 and GUI-360 without further experiments.

## Recommended paper structure

1. Introduction: diversity is not utility.
2. Related work: GUI self-improvement, trajectory refinement, verifiers, weak-to-strong supervision.
3. Counterfactual Revision Utility and Revision Utility Ladder.
4. Heterogeneous failure generation and global revision protocol.
5. Source–student reversal and prefix-consistency diagnosis.
6. Target×history factorial and negative-transfer controls.
7. Oracle utility-selection/replay ceiling.
8. Calibrated non-oracle ranker experiments.
9. Limitations and broader implications.
