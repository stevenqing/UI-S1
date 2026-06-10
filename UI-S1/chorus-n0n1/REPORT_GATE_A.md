# GATE A Report — Baseline Reproduction

**Status:** BLOCKED

## Scope

- `android_control` / `high_level` with `hiconagent_3b` from `/home/aiscuser/UI-S1/UI-S1/chorus-n0n1/configs/android_control_hiconagent.yaml`
- `gui_odyssey` / `test` with `hiconagent_3b` from `/home/aiscuser/UI-S1/UI-S1/chorus-n0n1/configs/gui_odyssey_hiconagent.yaml`
- `android_control` / `high_level` with `har_gui_3b` from `/home/aiscuser/UI-S1/UI-S1/chorus-n0n1/configs/android_control_har_gui.yaml`
- `gui_odyssey` / `test` with `har_gui_3b` from `/home/aiscuser/UI-S1/UI-S1/chorus-n0n1/configs/gui_odyssey_har_gui.yaml`

## Reproduction Metrics

Not run yet. Phase A is blocked until all prerequisites below are satisfied.

## Prerequisite Check

### android_control / high_level

- **OK** `benchmark_jsonl`: /home/aiscuser/UI-S1/UI-S1/datasets/android_control_evaluation_std.jsonl (8444 steps)
- **OK** `screenshots`: {"total_steps_checked": 8444, "missing_screenshot_examples": [], "screenshots_available": true}
- **MISSING** `model_checkpoint`: None
- **OK** `official_repo`: /home/aiscuser/UI-S1/UI-S1/chorus-n0n1/data/external/CVPR26-HiconAgent/AndroidControl_evaluation.py
- **OK** `api_url_configured`: http://localhost:8000/v1
- **OK** `api_port_open`: http://localhost:8000/v1
- **OK** `logprobs_requested`: True

### gui_odyssey / test

- **OK** `benchmark_jsonl`: /home/aiscuser/UI-S1/UI-S1/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl (25807 steps)
- **OK** `screenshots`: {"total_steps_checked": 25807, "missing_screenshot_examples": [], "screenshots_available": true}
- **MISSING** `model_checkpoint`: None
- **OK** `official_repo`: /home/aiscuser/UI-S1/UI-S1/chorus-n0n1/data/external/CVPR26-HiconAgent/Odyssey_evaluation.py
- **OK** `api_url_configured`: http://localhost:8000/v1
- **OK** `api_port_open`: http://localhost:8000/v1
- **OK** `logprobs_requested`: True

### android_control / high_level

- **OK** `benchmark_jsonl`: /home/aiscuser/UI-S1/UI-S1/datasets/android_control_evaluation_std.jsonl (8444 steps)
- **OK** `screenshots`: {"total_steps_checked": 8444, "missing_screenshot_examples": [], "screenshots_available": true}
- **OK** `model_checkpoint`: /home/aiscuser/UI-S1/UI-S1/chorus-n0n1/data/models/HAR-GUI-3B
- **OK** `official_repo`: /home/aiscuser/UI-S1/UI-S1/related_work/har/Inference/vllm_inference.py
- **OK** `api_url_configured`: http://localhost:8000/v1
- **OK** `api_port_open`: http://localhost:8000/v1
- **OK** `logprobs_requested`: True

### gui_odyssey / test

- **OK** `benchmark_jsonl`: /home/aiscuser/UI-S1/UI-S1/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl (25807 steps)
- **OK** `screenshots`: {"total_steps_checked": 25807, "missing_screenshot_examples": [], "screenshots_available": true}
- **OK** `model_checkpoint`: /home/aiscuser/UI-S1/UI-S1/chorus-n0n1/data/models/HAR-GUI-3B
- **OK** `official_repo`: /home/aiscuser/UI-S1/UI-S1/related_work/har/Inference/vllm_inference.py
- **OK** `api_url_configured`: http://localhost:8000/v1
- **OK** `api_port_open`: http://localhost:8000/v1
- **OK** `logprobs_requested`: True

## Blocking Issues

- [android_control] hiconagent_3b checkpoint is not configured locally (`model.checkpoint_path`).
- [gui_odyssey] hiconagent_3b checkpoint is not configured locally (`model.checkpoint_path`).

## Truncation Summary

- Total generations: 0
- Truncated generations: 0
- Truncation rate: 0.00%
- Note: no model generations have been run in preflight mode.

## Qualitative Step Records

No qualitative examples sampled because baseline inference has not run.

## Gate Decision

STOP. Do not proceed to N0. Resolve the blocking issues, run Phase A baseline reproduction, then regenerate this report.
