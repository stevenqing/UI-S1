# CARE: Counterfactual Acquisition with Risk-Controlled Evidence

Date frozen: 2026-08-11

Status: `FROZEN_AFTER_DIAGNOSTIC_BEFORE_CARE_RESULTS`

Boundary: CARE is a post-VUS exploratory method. VUS-SR, CEV-A, Utility-LSA, and all diagnostics in `headroom_audit.json` are known. CARE is not an independent confirmation; its final claim requires a new untouched benchmark.

## 1. First-principles problem

GUI test-time scaling is a sequential decision under a fixed candidate-generation budget, not a static voting problem.

After six shared proposals, the system has state $s_6$ and chooses one acquisition policy $a$ for the remaining six forwards. It then observes a 12-candidate set $C_a$, selects a real candidate $c$, or keeps a strong fallback $b$:

$$
s_6 \rightarrow a \rightarrow C_a \rightarrow \{c,\mathrm{KEEP}(b)\}.
$$

The objective is

$$
\max_{\pi_A,\pi_S}\; \mathbb E[R(\pi_S(C_{\pi_A(s_6)}))]
\quad\text{s.t.}\quad
\mathrm{cost}\le B,\;\mathbb E[R_{\pi}-R_b]\ge 0.
$$

For a fixed candidate set, success decomposes as

$$
P(R_{\hat c}=1)
=P(\exists c_i\in C:R_i=1)
\,P(R_{\hat c}=1\mid \exists c_i:R_i=1).
$$

The first term is acquisition coverage; the second is evidence-based identification. A safe override adds a third decision: whether evidence is strong enough to replace $b$.

## 2. Diagnostic facts that motivate CARE

All values below are descriptive, known before this protocol, and reproduced by `headroom_audit.py`.

### 2.1 Candidate identification dominates residual error

| Benchmark | VUS-SR safe | pass@12 | Candidate-ranking gap | Pairwise gate gap |
| --- | ---: | ---: | ---: | ---: |
| Mind2Web equal-arm | 34.92% | 59.21% | 18.52 pp | 5.77 pp |
| ScreenSpot-Pro equal-arm | 64.26% | 79.57% | 14.60 pp | 0.71 pp |

Here pairwise oracle chooses the better of VUS direct and CEV-A. Candidate-ranking gap is pass@12 minus that pairwise oracle. The dominant loss is finding the correct candidate, not adding model capacity to the existing gate.

### 2.2 Current visual input loses local evidence

Conditional ranking failure when at least one candidate is correct:

| Benchmark | Smallest target-area quartile | Largest quartile |
| --- | ---: | ---: |
| Mind2Web | 53.57% | 34.75% |
| ScreenSpot-Pro | 32.99% | 13.32% |

When exactly one of 12 candidates is correct, direct recall is 41.18% on Mind2Web and 8.37% on ScreenSpot-Pro. Full-screen downsampling plus overlapping labels is therefore insufficient for minority-truth candidates.

### 2.3 There is a real fixed-budget acquisition decision

The first six candidates are byte-equivalent at the public-record level across C-uni/C-cond/C-rand/C-self for every row. The remaining six differ. Oracle routing raises candidate coverage over the best static arm by 6.06 pp on Mind2Web and 3.67 pp on ScreenSpot-Pro. Thus acquisition can be learned without changing the 12-forward candidate budget.

## 3. Why ordinary verification is excluded

The frozen Q2b experiment achieved 73.68% binary verification accuracy but reduced final ScreenSpot B3 by 4.62 pp, 99% CI [−7.26,−2.14]. Its target was whether a crop contained the target center; filtering changed density and did not directly ask whether the marked candidate was evaluator-correct.

CARE therefore prohibits:

- independent YES/NO filtering;
- crop-presence labels as a proxy for candidate correctness;
- deleting candidates before set-level comparison;
- selecting a verifier by verifier accuracy rather than final Step-SR utility;
- broad low-reward actor negatives, which also harmed prior V23 pairwise training.

## 4. CARE method

### 4.1 A: counterfactual acquisition router

Input contains only information available after the six shared forwards:

- task and compact history;
- global screenshot representation;
- six candidate actions, parameters, coordinates, parse states, geometry, support, and cross-fitted reliability;
- no stage-2 candidate, stage-2 visual logit, arm outcome, or test label.

The router predicts four potential coverage outcomes

$$
q_a(s_6)=P(\exists c_i\in C_a:R_i=1\mid s_6),
$$

for `C_uni`, `C_cond`, `C_rand`, and `C_self`. All four outcomes are observed in the retained counterfactual bank, so this is full-information supervised policy learning, not off-policy bandit estimation.

Training targets are pass@12 coverage, not downstream VUS outputs. This prevents second-level stacking leakage. The deployed router chooses one arm, preserving six shared + six routed candidate forwards.

### 4.2 E: candidate-centric multi-scale evidence

Replace the full-screen A--L overlay with a single multi-image evidence query:

1. an unmarked global screenshot;
2. a fine candidate mosaic with 12 candidate-centered crops;
3. a context candidate mosaic with the same 12 candidates at a wider scale.

Each tile has one exact crosshair at the candidate point and a stable A--L label. The point is rendered at the tile center when in-frame; out-of-frame state is explicit. The legend gives action and parameter. No target box, positive DOM node, correctness label, source/model identity, or evaluator field enters the prompt.

Frozen crop scales are fractions of the shorter image edge: fine `0.07`, context `0.21`. Each crop is square and padded rather than warped. Duplicate coordinates retain separate action/parameter labels but may share image pixels.

The same retained Qwen3-VL-8B produces one-step A--L logits. This replaces the existing one-forward full-screen visual evidence query; it does not add a VLM forward to the post-acquisition selector.

### 4.3 R: relational counterfactual utility

For each challenger $i$ and fallback $b$, predict three mutually exclusive outcomes:

$$
y_{ib}\in\{\mathrm{REPAIR},\mathrm{SAME},\mathrm{BREAK}\},
$$

where REPAIR is $(R_i,R_b)=(1,0)$ and BREAK is $(0,1)$. The net utility is

$$
\Delta_{ib}=P(\mathrm{REPAIR})-P(\mathrm{BREAK}).
$$

The pair scorer is antisymmetric by construction:

$$
d(i,b)=h(e_i,e_b)-h(e_b,e_i).
$$

Training combines:

- listwise repair-or-KEEP cross entropy;
- pairwise Bradley–Terry loss only on discordant candidate/fallback pairs;
- fallback-correct BCE;
- a small U-GRPO expected-utility term.

No candidate is removed. The final candidate is selected from the full set using relational utility.

### 4.4 C: risk-controlled override

On calibration-only rows, order candidate thresholds from conservative to permissive. Use a one-sided grouped lower confidence bound on mean net utility. Choose the most permissive threshold whose bound is nonnegative, with family-wise control over the fixed threshold sequence.

The deployment rule is

$$
\mathrm{override}\iff \max_i L_{1-\alpha}(\Delta_{ib})>0,
$$

otherwise keep CEV-A. This optimizes final utility and directly controls break risk; binary verifier accuracy is not a selection metric.

## 5. Frozen experimental sequence

### Phase A0: audit

`headroom_audit.json` must satisfy all diagnostic gates D1--D5. This phase is already descriptive and cannot support a method claim.

### Phase A1: structural acquisition router

Train a permutation-equivariant six-candidate router with no new VLM inference. Five grouped outer folds and an inner checkpoint fold are required. Evaluate:

- selected-arm pass@12 versus nested best-static arm;
- selected-arm VUS-SR safe Step-SR versus nested best-static VUS-SR arm;
- regret captured relative to oracle arm routing.

Proceed only if pass@12 improves on at least one benchmark with 99% CI lower bound above zero and the other benchmark is noninferior within its MDE.

### Phase E0: frozen local-evidence anchor

Generate multi-scale mosaics on GPU 0--7 using fold-agnostic public inputs. Lock all logits and hashes before opening labels. Compare against the frozen full-screen visual anchor on:

- utility-positive candidate AUROC;
- direct recall on exactly-one-correct rows;
- direct recall in the smallest target-area quartile;
- final nested safe Step-SR.

Proceed if either benchmark gains at least 0.03 AUROC and the other loses no more than 0.01, or if nested safe Step-SR improves with positive 99% CI on one benchmark and no MDE loss on the other.

### Phase R1: relational set ranker

Only after E0 passes, train the fixed relational model using fold-sealed labels and atomic pretest records. Compare to VUS-SR, local-evidence zero-shot, CEV-A, and ordinary q2b-style filtering.

### Phase C1: calibrated risk control

Only after R1 passes, calibrate the fixed threshold sequence. Report repair/break counts, override coverage, one-sided utility bound, and Step-SR.

### Phase X: independent confirmation

Freeze CARE and evaluate without method changes on a third benchmark with a complete candidate bank. Until Phase X, CARE remains exploratory even if A1/E0/R1/C1 pass.

## 6. Main gates

| Gate | Requirement |
| --- | --- |
| CARE-1 | Selected-arm pass@12 beats nested best-static acquisition on one benchmark; other noninferior |
| CARE-2 | Relational evidence beats VUS-SR on Mind2Web with 99% CI lower bound positive |
| CARE-3 | All ScreenSpot cells remain noninferior to VUS-SR under 0.70 pp MDE |
| CARE-4 | Equal-benchmark standardized 99% CI positive versus VUS-SR |
| CARE-5 | Risk-control calibration lower bound nonnegative in every outer fold |
| CARE-6 | Frozen transfer improves an untouched third benchmark or is noninferior while reducing compute |

CARE becomes a confirmed method only if CARE-1--CARE-6 pass. CARE-1--CARE-5 without CARE-6 is a discovery result.

## 7. Kill conditions

- `CARE-K1`: first-six public candidates differ across arms;
- `CARE-K2`: any stage-2 field enters router input;
- `CARE-K3`: any target/evaluator field enters an evidence prompt;
- `CARE-K4`: local evidence fails E0;
- `CARE-K5`: relational model improves verifier accuracy but not final utility;
- `CARE-K6`: any outer-test label file opens before an atomic pretest selection record;
- `CARE-K7`: compute-matched extra-candidate control explains the gain;
- `CARE-K8`: independent benchmark is used for tuning rather than one-shot confirmation.

Only implementation defects permit rerun. Failed scientific gates do not authorize changing scales, prompts, losses, or thresholds.
