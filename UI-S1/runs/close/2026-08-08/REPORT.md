# Aggregator Closure and Native-Prompt Report

Date: 2026-08-08

Status: `COMPLETE_E_K1_TRIGGERED`

## 1. Executive conclusion

E1 answers the ownership question negatively. C-cond is the best arm by point estimate under majority on both benchmarks, and C-cond forms the global-best cell in both matrices. However, replacing the original aggregator with majority removes the statistically supported arm advantage:

- Mind2Web C-cond-majority minus C-uni-majority is +0.29 pp, 99% CI [-0.70,+1.26], below the 0.61 pp MDE.
- ScreenSpot-Pro C-cond-majority minus C-uni-majority is +1.27 pp, 99% CI [-0.14,+2.68]; it exceeds the 0.70 pp MDE but its CI lower bound is not positive.
- The C-rand and C-self controls also fail to retain positive 99% CI lower bounds under majority.

Therefore E-K1 triggers. The earlier cross-benchmark result remains a valid paired result under the frozen sequential aggregators, but it cannot be promoted to an aggregator-independent candidate-pool claim. The strongest defensible wording is:

> Cross-lineage consensus RoIs improve candidate use under specific type-first density aggregators; the improvement is not statistically distinguishable under exact-candidate majority voting.

Per the preregistration, E2 native-prompt inference and AndroidControl are cancelled. No additional GPU budget is spent to strengthen a restricted claim.

## 2. E1 design

E1 reuses all existing row-level traces and performs no new inference. Each benchmark is evaluated as a four-arm by seven-aggregator matrix:

- arms: C-uni, C-cond, C-rand, C-self;
- aggregators: majority, fold-held-out best slot (A0), original sequential aggregator, A1 geometric median, A2 density medoid, A3 joint PKA medoid, and A4 continuous PKA;
- priorities and A1 weights are fit only on non-test outer folds for each arm independently;
- Mind2Web uses website-fold-stratified episode bootstrap;
- ScreenSpot-Pro uses application-group bootstrap;
- all intervals use 10,000 resamples and 99% percentile bounds.

ScreenSpot-Pro contains a single implicit coordinate action. Majority therefore reduces to selecting the fold-development-priority exact candidate. B3 official is mapped to the A2 family only at the algorithm-family level: it uses complete-link grouping and coverage tie-breaking and is not identical to the Gaussian density medoid implemented as A2.

## 3. E1 Mind2Web

The global-best cell is C-cond + majority at 32.31%. The row best is majority for C-uni, C-cond, and C-self, while C-rand is best under A0. The best arm differs by aggregator: C-cond is best under majority, A1-A4, and the original sequential aggregator; C-uni is best under A0 because C-uni and C-cond tie at 31.88% and the frozen tie-break selects C-uni.

The key distinction is between point ranking and statistical closure. C-cond-majority has the highest point estimate, but its +0.29 pp advantage over C-uni-majority is below MDE and has a CI crossing zero. C-cond also fails both mandatory majority controls.

This confirms arm–aggregator interaction. The large +4.90 pp C-cond effect under the original sequential aggregator does not survive replacement by majority; under majority the same arm change is only +0.29 pp.

## 4. E1 ScreenSpot-Pro

The global-best cells are C-cond + A2 and C-cond + A3, tied at 66.48%. C-cond is the best arm for every aggregator. Aggregator row optima vary:

- C-cond: A2/A3;
- C-uni and C-self: A4;
- C-rand: official B3.

Thus C-cond is structurally strong on ScreenSpot-Pro, but the preregistered majority claim is not closed. Majority gives C-cond a +1.27 pp point advantage over C-uni and C-rand, but both CIs have a slightly negative lower bound. Against C-self, the gain is only +0.44 pp and below MDE.

The original B3 comparison remains valid: C-cond exceeds C-uni by +2.21 pp with 99% CI [+0.50,+4.16]. E1 shows that this result belongs jointly to the C-cond arm and the official B3 aggregation semantics, not to C-cond alone.

## 5. Aggregator ownership

E1 establishes three separate facts.

1. **Arm ranking is not arbitrary.** C-cond is the global-best arm by point estimate on both benchmarks.
2. **The inferential claim is aggregator-dependent.** Majority does not preserve the positive-CI arm effect.
3. **The original aggregators are not merely weak baselines.** On ScreenSpot-Pro, A2/A3 slightly exceed B3 for C-cond; on Mind2Web, majority exceeds the original sequential aggregator. The observed Q1 gains arise from a compatible arm–aggregator pair rather than a universally dominant aggregation rule.

The main table must therefore present the full matrix or, at minimum, show C-cond under both the original aggregator and majority. Reporting only 65.91% and 31.59% as globally best systems would be misleading.

## 6. E2 and SOTA line

E2 is cancelled by E-K1 before any native-prompt inference. `e2_native_prompt.json` records `CANCELLED_NOT_RUN`.

The SOTA line remains open. The unified product-action prompt preserves internal paired validity across arms, but it lowers full-image model performance by roughly 20 pp relative to historical native-adapter results. Because the old row-level native traces are missing and native anchors were not rerun, the current absolute values cannot support a SOTA claim.

Valid writing must explicitly state:

- all four Mind2Web arms share the same preregistered unified prompt, so internal arm comparisons are valid;
- absolute scores are below historical native-adapter values;
- no claim against published Mind2Web SOTA is made;
- native-prompt reruns were preregistered but cancelled after E-K1 restricted the method claim.

## 7. E3 containment mechanism

E3 supports the high-start condition on the frozen qualitative criterion.

ScreenSpot-Pro starts with near-complete rank-0 containment (99.94%) and loses 38.90 pp by rank 11. Its V-only performance falls significantly from N4 to N16 by 2.91 pp, with 99% CI [-5.58,-0.36].

Mind2Web starts much lower at 40.38% and loses only 9.23 pp by rank 11. Its V-only N16-minus-N4 change is +0.34 pp with 99% CI [-1.22,+1.98], statistically indistinguishable from zero.

All four preregistered qualitative conditions pass. The appropriate interpretation is:

> Geometric rank decay is present on both benchmarks, but a visible negative performance slope requires a high-quality early proposal regime with substantial quality to lose.

This converts XF4 from a simple cross-benchmark contradiction into support for a benchmark-dependent high-start condition. The evidence contains only two benchmark points, so it is not a fitted relationship or universal law. E-K4 does not trigger.

## 8. AndroidControl

AndroidControl was paused with row-level fsynced checkpoints before E1:

- UI-AGILE Low/High: 2,000 / 2,000;
- GUI-R1 Low/High: 1,096 / 1,056;
- UI-R1-E Low/High: 1,824 / 1,792.

E-K1 permanently cancels these lanes under the current closure protocol. They are not resumed, scored, or used in any claim. External PID 2274 was not modified.

## 9. Final claim boundary

Supported:

- C-cond plus the frozen sequential density aggregators significantly beats C-uni, C-rand, and C-self on both ScreenSpot-Pro and Mind2Web.
- C-cond is the point-estimate-best arm under majority on both benchmarks.
- The high-start containment condition qualitatively explains why ScreenSpot-Pro shows V-only decline while Mind2Web does not.

Not supported:

- C-cond candidate pools are aggregator-independently superior.
- The current system beats majority voting.
- The current Mind2Web result reaches SOTA.
- Rank decay is a universal budget-performance law.

## 10. Artifacts

- `configs/aggregator_map.yaml`
- `configs/native_adapters.yaml`
- `configs/e3_mechanism.yaml`
- `e1_arm_aggregator_matrix.json`
- `e2_native_prompt.json`
- `e3_containment_mechanism.json`
- `fig_containment.pdf`
- `MAIN_TABLE.md`
- `AC_PAUSE_STATUS.json`
