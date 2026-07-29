# Gate 1 Report: SeeClick on Mind2Web Cross-Task

Verdict: **PASS - STOPPED FOR REVIEW**

Gate 1 reproduces the released SeeClick Mind2Web Cross-Task result on one visible
NVIDIA A100. AndroidControl has not been installed or run.

## Result

All values below are task/episode-macro percentages except parse rate.

| Result | Steps | Element Accuracy | Operation F1 | Step SR | Parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper Table 4 anchor | - | 28.3000 | 87.0000 | 25.5000 | - |
| Native upstream traversal | 2,079 | 28.5535 | 87.0421 | 26.0095 | 100% |
| Corrected full coverage | 2,080 | 28.5204 | 86.9429 | 25.9765 | 100% |
| Corrected delta vs anchor (pp) | - | +0.2204 | -0.0571 | +0.4765 | - |

The corrected result is within 1 percentage point of the exact live-paper row
on all three primary metrics. The independent audit exits successfully and
finds zero duplicate images, zero recorded-vs-recomputed score mismatches, and
zero native-summary deltas above `1e-12`.

## Count Diagnosis

The Cross-Task annotations contain 2,094 actions: 2,080 have a bbox and 14 do
not. The unmodified evaluator traversal produced 2,079 predictions because one
scoreable TYPE action contains `32" curved monitor`. Upstream `action2step`
interpolated that value into a quoted Python-literal string without escaping;
`ast.literal_eval` raised `SyntaxError`, and a broad `except` silently skipped
the action before inference.

The evaluator was changed to construct the GT dictionary from annotation fields
instead of parsing its own string. The missing action was the first step of its
episode, so it had no previous-action dependency and was replayed exactly once
with `--annotation_id ... --max_steps 1`. Its raw model output was CLICK at
`(0.03, 0.06)` against TYPE GT, yielding Element=0, Operation F1=0, Step=0.
Native artifacts remain untouched; the corrective row is stored separately.

## Scope And Semantics

- Data: SeeClick-processed screenshots and annotations, Cross-Task split.
- Model: `cckevinn/SeeClick-mind2web` at pinned revision.
- Prompt/history: upstream prompt and last-four-action history unchanged.
- Element: inclusive normalized predicted-point-in-GT-bbox.
- Operation: upstream token-set F1 over action type and TYPE/SELECT value.
- Step: Element correct and Operation F1 exactly 1.
- Macro: mean of per-episode means over all 252 episodes.
- Device: exactly one visible GPU, enforced by the evaluator.

The spec's approximate 78.4 Element Accuracy does not match the live SeeClick
paper's Cross-Task row. The valid same-split, same-model Table 4 anchor is
28.3 Element Accuracy, 87.0 Operation F1, and 25.5 Step SR. The evaluator's
category-averaged operation statistic is not used for paper comparison.

## Artifacts

- Native raw output: `artifacts/gate1_cross_task/predictions.jsonl`
- Corrective raw output: `artifacts/gate1_corrected_missing/predictions.jsonl`
- Independent audit: `artifacts/gate1_audit.json`
- Audit implementation: `audit_gate1.py`
- Source and model provenance: `provenance.json`
- Data integrity and coverage: `data_manifest.json`
- Locked environment: `uv.lock` and `env.txt`

Per the fail-closed execution order, this run stops here pending explicit Gate 1
review. No Gate 2 command is queued or running.