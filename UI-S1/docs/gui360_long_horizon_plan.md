# GUI-360 Long-Horizon Implementation Plan

Source: user-provided "GUI-360 Long-Horizon - Implementation Specs for a Coding Agent" in the Copilot transcript.

## Build Order

1. `data/loader.py` smoke test.
   - Status: implemented.
   - Acceptance: raw JSONL only, no `processed_data`, success GT populated, fail GT absent, tar/direct images open.

2. `data/difficulty.py` plus `validity_gate`.
   - Status: implemented as code and synthetic tests.
   - Remaining runtime gate: run calibrated difficulty with held-out `Qwen2.5-VL-72B`, then validate against model-under-test errors.

3. `data/divergence.py` plus audit.
   - Status: implemented as deterministic/pluggable embedding scaffold.
   - Acceptance: success manifold, `t*`, `delta`, tau calibration, audit bundle and agreement guard.

4. `recovery_oracle.py` plus audit.
   - Status: implemented as high-precision rule scaffold.
   - Acceptance: content-independent targets only; precision audit API enforces threshold.

5. `harness/model.py`, `prompt.py`, `predict.py`, `correctness.py`.
   - Status: implemented.
   - Acceptance: GT-absent correctness raises, cache purity, visual-a11y prompt changes inputs.

6. `experiments/` pure runners emitting `Row`.
   - Pending modules: `m_textmem`, `m_core`, `m_plan`, `m_textdrift`, `m_recover`, `m_diag`.

7. `analysis/` decision layer and guards.
   - Pending modules: `stats.py`, `identifiability.py`, `controls.py`.

8. `configs/` and `run_all.py` orchestrator.
   - Pending: YAML config and short-circuiting stage runner.

## Hard Invariants

- Off-manifold/content-absent rows must keep `step_correct=None`.
- `step_correct` must raise on GT-absent steps.
- Difficulty must not be computed from the model under test.
- `m_diag` is non-identifying and may expose only an upper bound, never a causal drift estimate.
- Recovery oracle targets must be content-independent; rules that require document/cell content return `None`.
- Headline effects are reportable only after shuffle/audit guards pass.
