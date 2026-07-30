# Complementarity Follow-up Report

Date: 2026-07-30

Status: `PARTIAL` pending E4 human labels and ten E5 inference cells.

## Scope and data contract

The shared artifact contains 102,054 `(row identity, model)` rows:

| Pool | Tidy rows | Unique identities | Models |
|---|---:|---:|---:|
| AndroidControl Low | 38,540 | 7,708 | 5 |
| AndroidControl High | 38,540 | 7,708 | 5 |
| Mind2Web visual | 22,880 | 2,080 | 11 |
| Mind2Web HTML | 2,094 | 2,094 | 1 |

The 58 AndroidControl parameter-conflict identities are marked on both settings and all five models, producing 580 quarantined tidy rows. Downstream code excludes them by default.

`scoring.py` was extracted from the upstream analyzers. Regenerating both upstream summaries after extraction produced byte-identical files:

- AndroidControl SHA256: `5c4e9495c1b1eaaee46fad7101ef148174bb41d1cf51c5b99b4931ff2188adfb`;
- Mind2Web SHA256: `f0418b3f42806ac6026cadc7e35ad4c948128256c20294798e1e3ff21baa6609`.

AndroidControl source parquet has no app or episode identifier and its `group` field is constant `android`. The grouped folds therefore use the SHA256 of the released High full instruction as a conservative task/goal group and map Low rows through the audited cross-setting identity. This yields 1,393 groups. It is task-held-out, not app-held-out, and must not be described as app-held-out evaluation. Mind2Web uses 69 website groups.

## E1: training-free geometric ensemble

Five shared grouped folds select model subsets and tie priorities on the four dev folds and evaluate once on the held-out fold.

| Pool | Best single | Stage A | A+B | A+B+C | Weighted full | Delta | Full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| AndroidControl Low | 79.11% | 79.11% | 79.11% | 79.11% | 79.11% | 0.00 pp | 87.54% |
| AndroidControl High | 60.76% | 60.76% | 60.76% | 60.76% | 60.76% | 0.00 pp | 77.53% |
| Mind2Web visual | 51.78% | 51.95% | 54.10% | 54.10% | 54.32% | +2.55 pp | 81.78% |

AndroidControl dev selection retained only the best single model in every fold: UI-AGILE-3B for Low and UI-AGILE-7B for High. Therefore action voting, geometric dispersion, and vote margin are constants, and all three confidence AUROCs are exactly 0.5 by construction. This is a negative ensemble result, not a confidence implementation failure.

Mind2Web gains come from Stage B geometric aggregation rather than action plurality: Stage A adds only 0.17 pp, while A+B adds approximately 2.32 pp. Weighted geometric median reaches +2.55 pp. The confidence AUROCs are:

- vote margin: 0.548;
- negative geometric dispersion: 0.660;
- logistic combination: 0.623.

The selected Mind2Web subset is unstable: three folds select five models, while two folds retain a single TongUI model. The +2.55 pp result is therefore promising but not yet a deployment recommendation. E1's formal close/continue decision remains pending E5's MDE.

## E2: router feasibility upper bound

Status: `PASS` for the experiment; decisions are pool-specific.

The implementation uses three separate disagreement pools, the shared grouped folds, one gradient-boosted binary head per model, and no GT-derived features:

- T0: instruction hashing embedding, history length, and low-dimensional screenshot statistics;
- T1: T0 plus every model's released-parser action/coordinate/parse/raw-output features;
- T2: T1 plus E1 vote margin, geometric dispersion, and pairwise action-agreement summaries.

Both direct routing and dev-tuned abstention-to-best-single are evaluated. The routing line is not closed until T2 test-fold headroom capture is available.

The outer test folds are untouched. Within each outer split, three folds train the binary heads, one fold selects best-single and tunes the abstention threshold, and one fold is evaluated once.

| Pool | T0 pooled capture | T1 pooled capture | T2 pooled capture | T2 projected full-pool delta | T2 abstain capture |
|---|---:|---:|---:|---:|---:|
| AndroidControl Low | 14.11% | 29.92% | 29.92% | +2.52 pp | 30.08% |
| AndroidControl High | -2.03% | 5.69% | 5.69% | +0.95 pp | 5.53% |
| Mind2Web visual | 4.57% | 15.75% | 14.65% | +4.47 pp | 9.45% |

AndroidControl High fails the preregistered 10% T2 gate, so routing is closed for that pool and the next line is distillation/model merging. Low and Mind2Web pass the upper-bound gate, but only after using all model outputs. They are therefore multi-model rerankers rather than input-only routers and must be compared to equal-compute E1.

T2 adds no signal over T1 in either AndroidControl pool because E1 selected only one model, making consistency features constant. On Mind2Web, T2 is worse than T1 (14.65% vs 15.75% capture), and abstention further reduces capture. The output feature block carries nearly all permutation importance in T1. This does not support the hypothesis that E1 consistency features are the missing routing signal.

## E3: corrected oracle curves

All values below exclude quarantine. Oracle values are descriptive upper bounds.

| Pool | Full oracle micro | Full oracle episode-macro | Deployable oracle micro | Deployable models | Models for 95% of full oracle |
|---|---:|---:|---:|---|---:|
| AndroidControl Low | 87.54% | 88.18% | 87.54% | all five | 2 |
| AndroidControl High | 77.53% | 77.98% | 76.27% | GUI-R1-3B/7B, UI-AGILE-3B/7B | 3 |
| Mind2Web visual | 81.78% | 83.97% | 70.91% | TongUI-3B/7B/32B, UI-TARS-7B | 5 |

Deployable means parse failure rate below 5% and step-micro above 30%. The deployable Mind2Web oracle is 10.87 pp below the eleven-model oracle. The eleven-model number therefore belongs in the appendix; the 70.91% deployable micro oracle is the defensible main-text upper bound.

Greedy Mind2Web selection starts with TongUI-7B (1,101 successes), adds CogAgent-18B (+336), TongUI-32B (+116), and UI-TARS-72B (+61). Five models are needed to reach 95% of full-oracle successes. The previous `all 11 succeed = 0` observation is removed because it is mechanically constrained by the weakest model.

Mind2Web aggregation is now explicit: step-micro averages 2,080 identities, while episode-macro first averages within each of 252 episodes. Full-oracle micro and macro are 81.78% and 83.97%, respectively; they are not interchangeable.

## E4: hard-core audit

Status: `READY_FOR_HUMAN_ANNOTATION`.

The deterministic package contains:

- 100 stratified AndroidControl Low hard-core rows;
- 50 stratified AndroidControl High control rows;
- 100 stratified Mind2Web visual hard-core rows;
- all 79 Mind2Web SELECT rows for option-visibility labeling;
- all 109 MindAct-only rows as cross-modal audit context.

Thirty main-audit rows are assigned to both annotators. `audit_report.py` refuses to run while any assigned label is null and requires Cohen kappa at least 0.6. No E4 proportions, Wilson intervals, corrected learnable ceiling, or Tier-4 curriculum decision are claimed before two real annotators complete the package.

## E5: noise floor

Status: `PENDING_INFERENCE`.

The preregistered factorial is three prompt variants by two visual profiles, greedy decoding, and no seed dimension. Existing audited traces provide the original-prompt/original-profile cell. Ten cells remain:

- AndroidControl GUI-R1-7B High: original processor profile is the verified 12,800-token default; deployment profile is 768 tokens;
- Mind2Web TongUI-7B: original profile is the verified 1,344-token budget; deployment profile is 768 tokens.

Both original profiles reproduce an existing row-0 input hash or prompt/resize provenance exactly. MDE will be twice the sample SD across six cells. No MDE or transition-significance claim is currently available because GPUs 0--3 are occupied by existing workloads.

## Diagnostics

### D1: wait attribution

Every model predicts `wait` more often in High than Low. The clearest case is GUI-R1-3B: predicted-wait base rate rises from 0.84% to 8.42%, while precision falls from 82.8% to 25.9%. UI-AGILE-3B rises from 1.23% to 5.99%, with precision falling from 95.7% to 63.1%.

Thus the apparent High improvement on GT=`wait` is partly consistent with increased wait hedging. The previous interpretation as state/history understanding is withdrawn unless a controlled analysis separates these effects.

### D2: SELECT visibility

All 79 SELECT rows are included in the human package. Until option visibility is labeled, the 53/79 visual hard-core count must not be treated as a clean curriculum set.

### D3: chance-corrected failure overlap

Failure Jaccard is replaced by Cohen kappa with 1,000 matched-marginal permutations.

- AndroidControl Low: UI-AGILE-3B/7B kappa 0.784 and GUI-R1-3B/7B kappa 0.774, both permutation `p < 0.001`.
- AndroidControl High: UI-AGILE-3B/7B kappa 0.621 and GUI-R1-3B/7B kappa 0.609, both `p < 0.001`.
- Mind2Web: TongUI-3B/7B is strongest at kappa 0.587, `p < 0.001`.
- Qwen2.5-VL-3B/ShowUI-2B is kappa 0.003, `p = 0.458`; Qwen2.5-VL-3B/UI-TARS-2B is kappa -0.008, `p = 0.894`.

The earlier Qwen 3B/7B Jaccard of 0.946 was dominated by marginal failure rates and is not evidence of meaningful shared behavior.

### D4: grounding-threshold sensitivity

Sweeping the AndroidControl radius from 0.06 to 0.30 does not change model ordering in either setting; Kendall tau against the 0.14 ranking is 1.0 throughout. Oracle Step SR rises from 85.37% to 89.36% in Low and from 73.67% to 81.74% in High. Absolute results are threshold-sensitive, but the comparative ranking is not.

### D5: High-only jitter test

Observed High-only counts are below the matched-marginal independence expectation for every model: observed/expected ranges from 0.211 to 0.472. Among High-only rows with a Low grounding distance, 27%--41% fall in `[0.14, 0.28)`, but none of the five models differs significantly from all Low grounding failures by Mann-Whitney test (`p = 0.309`--`0.930`).

Therefore the data do not support the claim that High-only successes are mainly threshold jitter. They also do not establish history-conditioned correction: positive cross-setting dependence and action/text transitions remain alternative explanations. The earlier history-conditioned wording is withdrawn.

## Decision state

| Line | Current evidence | Decision |
|---|---|---|
| Geometric ensemble, AndroidControl | 0.00 pp over best single in Low and High | Negative, formal close pending MDE |
| Geometric ensemble, Mind2Web | +2.55 pp; dispersion AUROC 0.660 | Continue only if gain exceeds MDE |
| Routing, AndroidControl High | T2 pooled capture 5.69% | Close; move to distillation/model merging |
| Routing, AndroidControl Low | T2 pooled capture 29.92%, requires all outputs | Upper bound passes; equal-compute evaluation required |
| Routing, Mind2Web | T1/T2 pooled capture 15.75%/14.65%, requires all outputs | Upper bound passes; consistency adds no benefit |
| Full eleven-model oracle | 81.78% micro | Appendix only |
| Deployable Mind2Web oracle | 70.91% micro | Main-text upper bound |
| Hard-core curriculum Tier 4 | Human audit incomplete | No decision |
| Transition significance | E5 incomplete | No decision |

## Reproduction

```bash
.venv-ac-vllm/bin/python runs/complementarity/2026-07-30/build_rows.py \
  --output runs/complementarity/2026-07-30/rows.parquet \
  --manifest runs/complementarity/2026-07-30/rows_manifest.json \
  --folds runs/complementarity/2026-07-30/folds.json

.venv-ac-vllm/bin/python runs/complementarity/2026-07-30/e1_ensemble.py \
  --output runs/complementarity/2026-07-30/e1_ensemble.json

.venv-ac-vllm/bin/python runs/complementarity/2026-07-30/e2_router.py \
  --e1 runs/complementarity/2026-07-30/e1_ensemble.json \
  --output runs/complementarity/2026-07-30/e2_router.json

.venv-ac-vllm/bin/python runs/complementarity/2026-07-30/e3_oracle.py \
  --output runs/complementarity/2026-07-30/e3_oracle.json \
  --table runs/complementarity/2026-07-30/oracle_table.md

.venv-ac-vllm/bin/python runs/complementarity/2026-07-30/diagnostics.py \
  --output runs/complementarity/2026-07-30/diagnostics.json

runs/complementarity/2026-07-30/launch_e5.sh all
```