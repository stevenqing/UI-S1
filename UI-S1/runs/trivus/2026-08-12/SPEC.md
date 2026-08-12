# TriVUS: Three-Benchmark Variable-Set Utility Selection

Date frozen: 2026-08-12

Status: `PREREGISTERED_BEFORE_ANDROIDCONTROL_RECOVERY`

## 1. Scope

TriVUS is a new exploratory optimization study restricted to Mind2Web, ScreenSpot-Pro, and AndroidControl. All VUS-SR, CARE, RAVEL, DELTA, CIVA, and historical AndroidControl results are known. No claim of untouched external confirmation is allowed; the target is the strongest auditable method on these three benchmark families.

The study tests one first-principles hypothesis: a fallback-relative utility selector should operate on an unordered set of valid actions rather than a fixed candidate count. A shared model can then learn action, geometry, disagreement, and blind visual evidence across $K=12$ desktop/web candidate sets and $K=3$ AndroidControl model pools without inventing fake candidates or exposing model identity.

## 2. Stage R0: result-blind AndroidControl recovery

The existing AndroidControl three-model intersections contain 1,096 Low and 1,056 High rows and have 2--3 pp subset bias. They cannot carry a primary method claim.

R0 resumes the exact frozen stage-1 lanes to 2,000 rows per setting:

| Lane | Existing | Missing | Final |
| --- | ---: | ---: | ---: |
| GUI-R1-7B Low | 1,096 | 904 | 2,000 |
| GUI-R1-7B High | 1,056 | 944 | 2,000 |
| UI-R1-E-3B Low | 1,824 | 176 | 2,000 |
| UI-R1-E-3B High | 1,792 | 208 | 2,000 |

UI-AGILE-7B is already complete at 2,000/2,000. Recovery uses the original model revisions, official prompts/parsers, temperature zero, 256 output tokens, processor pixel bounds, row order, single-shard identity, and per-row fsync. Existing partial files are copied to a new run directory and never modified in place.

R0 is result-blind: no candidate success, accuracy, oracle, aggregator, or benchmark comparison may be computed until all four recovered lanes pass exact identity and provenance checks and are hash-locked.

## 3. Unified action-set bank

After R0 passes, construct a new public/private interface.

Public fields:

- benchmark and setting/arm;
- row ID, episode/group, frozen fold;
- instruction and compact history;
- image path and SHA-256;
- $K$ candidates with action type, normalized coordinate when applicable, parameter, and parse state;
- candidate mask and candidate count.

Private fields contain only benchmark-specific candidate-success bits and fallback-success bits.

Candidate sets:

- Mind2Web and ScreenSpot-Pro: the existing four 12-candidate arms;
- AndroidControl Low and High: UI-AGILE-7B, GUI-R1-7B, and UI-R1-E-3B predictions, yielding $K=3$.

Source/model/slot identity is prohibited from public model inputs. Candidate order is independently hash-permuted per row and epoch. AndroidControl Low and High versions of the same source row use the same episode fold, preventing paired-image leakage across outer folds.

Benchmark-specific evaluators are permitted only while constructing private labels:

- Mind2Web exact action/parameter/evaluator contract already frozen in the candidate bank;
- ScreenSpot-Pro coordinate contract already frozen in the candidate bank;
- AndroidControl action type, coordinate radius 0.14, and text matching from the frozen shared scorer.

## 4. Blind visual evidence

Mind2Web and ScreenSpot-Pro reuse the locked fallback-agnostic VUS label logits. AndroidControl receives one new blind Qwen3-VL-8B selector call per row/setting over the three hash-permuted actions, with labels A--C only.

The prompt may include screenshot, instruction, compact history, action, normalized coordinate, and parameter. It may not include source identity, fallback, reliability, ground truth, evaluator state, target box, candidate success, fold, or prior benchmark result.

The AndroidControl logits and public rows must be hash-locked before any private label is joined.

## 5. Variable-set model

TriVUS uses one shared permutation-equivariant masked set encoder:

- shared candidate encoder;
- explicit candidate mask for $K\in\{3,12\}$;
- no positional embedding or source/slot embedding;
- one learned KEEP representation derived from the frozen fallback candidate/state;
- candidate/KEEP utility head;
- fallback-correct downside head.

Targets remain fallback-relative:

- if one or more candidates repair an incorrect fallback, distribute listwise mass over repair candidates;
- otherwise target KEEP;
- train fallback correctness and expected standardized utility as auxiliary objectives.

Frozen fallbacks:

- Mind2Web/ScreenSpot-Pro: exact nested CEV-A policy;
- AndroidControl: fold-local action plurality with dev-reliability candidate tie-break, matching the strongest auditable aggregation family.

## 6. Required variants and controls

- `JOINT3`: shared model trained with equal family mass across Mind2Web, ScreenSpot-Pro, and AndroidControl;
- `TARGET_ONLY`: independent same-capacity model per benchmark family;
- `JOINT2_NO_ANDROID`: shared Mind2Web/ScreenSpot-Pro model;
- `NO_VISUAL`: JOINT3 without blind selector logits;
- `SOURCE_ID_PLACEBO`: source identity replaced by a row-wise random permutation and used only as a negative control; true source identity remains prohibited;
- frozen VUS-SR on Mind2Web/ScreenSpot-Pro;
- fold-local majority and best-single UI-AGILE on AndroidControl;
- random candidate and frozen fallback controls.

No variant is selected after outer-test access. `JOINT3` is primary.

## 7. Nested protocol

- five grouped outer folds shared by all benchmark families;
- AndroidControl Low/High paired rows remain in the same fold;
- within each outer fold: two model-training folds, one checkpoint fold, one OOF threshold fold;
- equal benchmark-family weight; AndroidControl Low/High split the AndroidControl family mass;
- train-only feature normalization;
- physical per-fold private-label files;
- atomic fsynced pretest record before any outer-label access;
- one outer evaluation;
- 10,000 paired grouped bootstrap resamples with 99% percentile intervals.

Mind2Web and ScreenSpot-Pro retain MDEs 0.006106589385659482 and 0.007. AndroidControl Low/High use a preregistered 0.01 practical noninferiority margin.

## 8. Stage gates

R0 passes only if all six AndroidControl lanes have exactly 2,000 unique rows, exact reference identity, zero provenance mismatch, and locked hashes.

R1 permits blind selector inference only if, on the complete 2,000-row lanes, the three-candidate oracle exceeds fold-local majority by more than 1.0 pp on both Low and High. This diagnostic is computed once from locked private labels.

TriVUS is promoted only if all conditions hold:

1. every Mind2Web and ScreenSpot-Pro arm is noninferior to frozen VUS-SR using the benchmark MDE;
2. AndroidControl Low and High are each noninferior to fold-local majority within 1.0 pp;
3. at least one benchmark family has a positive equal-cell 99% CI versus its frozen baseline;
4. the three-family standardized 99% CI versus frozen baselines is positive;
5. JOINT3 has a positive three-family standardized 99% CI versus TARGET_ONLY;
6. JOINT3 has a positive three-family standardized 99% CI versus NO_VISUAL;
7. no benchmark family loses more than one MDE versus its strongest frozen baseline.

## 9. Kill conditions

- `T-K1`: incomplete or mismatched AndroidControl recovery;
- `T-K2`: AndroidControl three-candidate oracle headroom at most 1.0 pp in either setting;
- `T-K3`: any candidate source/slot identity, target, evaluator, or label enters public inputs;
- `T-K4`: any outer label opens before pretest fsync;
- `T-K5`: JOINT3 gain is explained by TARGET_ONLY capacity or NO_VISUAL structure;
- `T-K6`: any benchmark family loses more than its frozen MDE;
- `T-K7`: architecture, loss, candidate set, prompt, threshold, or gate changes after formal results.

R0 completion authorizes only the locked-bank audit. R1 completion authorizes blind AndroidControl selector inference. Neither authorizes model training until a result-free exact-training amendment and implementation commit exist.