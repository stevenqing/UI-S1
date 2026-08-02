# Diversity-Axis Execution Spec

Date requested: 2026-08-02

Freeze date: 2026-08-01

Upstream:

- `runs/ccm-h2h/2026-07-31/`
- `runs/allocation-law/2026-08-01/` at `959aec122c1b3d1d77fdcaf9d3ae4f355146da83`

Status: result-free preregistration. No X1-X8 result has been computed in this workdir.

## Primary questions

- X1: whether same-model sampling has a negative budget slope under the official GUI-RC aggregator.
- X2: whether adaptive zoom and cross-lineage allocation compose at fixed useful-forward budget.
- X3: whether the L1 slope signs survive application-group bootstrap and area stratification.
- X4-X7: functional-pipeline baseline, topology, unlabeled pool ranking, and confidence diagnostics.
- X8: a Mind2Web full-view lineage-only alternative that does not alter blocked L3.

## Result-free preflight findings

### X1

The official GUI-RC source is `ZJU-REAL/GUI-RCPO` commit `af15ed5fe8b89b0fe5043a3e94f2984c7b126a4b`, file blob `20a577695c488ea5a75fe685dabf9e5bc1d50757`. It expands point predictions to 50x50 pixel boxes, accumulates a per-pixel vote grid, extracts connected components of the maximum-vote mask, selects the largest component, and returns its bounding-box center. Ties retain the first component in scan order because the implementation updates only on strictly larger area.

The official reported protocol uses 64 samples, temperature 0.5, and top-p 0.95. The local GTA1 trace has exactly five raw samples per identity at temperature 0.7 and no top-p field. It can test the port at N=4 only; it cannot support the preregistered N=4/8/12/16 slope. X1 must remain blocked until additional result-blind sampling generation is frozen and completed. Existing N=5 is never repeated or padded.

The supplied sanity text conflates rows in the official table: Qwen2.5-VL-3B improves 80.11 to 82.63 on ScreenSpot-v2, while 83.57 is OS-Atlas with GUI-RC. The implementation uses the corrected anchor.

### X2

UI-Zoomer is selected because the official implementation is available and complete. Exact source and parameters are frozen in `configs/zoom_method.yaml`.

The official method uses eight stochastic global samples and one conditional deterministic crop refinement. Its useful-forward count is therefore eight on reliable rows and nine on uncertain rows, while Q1/Q3 each use twelve candidates. This is a real design incompatibility, not an implementation detail. X2 inference is blocked until a result-free amendment defines equal useful-forward accounting without duplicate padding or ignoring refinement cost.

### X3

All required raw candidates exist. Per-row held-out B3/M1 outputs can be reconstructed deterministically with the frozen Allocation-Law loaders and `evaluate_pool`. Bootstrap uses the fold-stratified application resampling fixed by Allocation-Law Amendment 004. N=24 remains one-sided and is excluded from two-curve slope inference.

### X8

Mind2Web has six deployable full-view models but no family has six deployable members: TongUI has three, UI-TARS has two, and CogAgent has one. Therefore an equal-budget N=6 same-lineage versus cross-lineage full-view comparison cannot be constructed without padding, non-deployable models, or view reuse. X8 remains blocked unless a result-free amendment changes the estimand; blocked L3 is not modified.

## Fail-closed rules

- No candidate duplication or padding.
- No target field enters proposal or inference paths.
- No method, threshold, model, pool, or budget operation changes after its target outcome is inspected.
- Official ports pin upstream commit/blob and preserve tie behavior.
- Unavailable comparisons are reported as unavailable, not silently replaced.
- `download_models.py` is unrelated user work and is not included.
