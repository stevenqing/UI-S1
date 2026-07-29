# Mind2Web Gate 1: SeeClick

Status: `GATE_1_PASS_STOPPED_FOR_REVIEW`

This run is limited to the SeeClick reproduction gate. AndroidControl and all
downstream baselines remain blocked until this gate has a reviewed result.

## Fixed evaluation contract

- Input source: SeeClick-processed Mind2Web screenshots and annotations.
- Split: Cross-Task (`--task task`).
- Model: `cckevinn/SeeClick-mind2web`.
- Base tokenizer/config: `Qwen/Qwen-VL-Chat`.
- Device policy: one GPU only; no `device_map=auto`.
- Prediction: normalized point `(x, y)` in `[0, 1]`.
- Element correctness: inclusive point-in-ground-truth-bbox.
- Operation: action type plus token F1 for `TYPE`/`SELECT` values.
- Reported gate metrics: macro Element Accuracy, macro Operation F1, macro Step SR.
- Parse fairness: parse success count and rate must accompany the metrics.
- Any input, count, parser, metric, or anchor mismatch is a STOP and requires a
  written diagnosis. No silent adjustment is permitted.

## Environment deviation

Upstream `requirements_agent.txt` pins `torch==1.12.1+cu116`. This run uses
`torch==2.1.2` with the same upstream `transformers==4.36.2` and
`peft==0.7.1`, because the host uses a modern CUDA driver and `uv` resolves an
isolated Python 3.10 environment. The model/evaluator smoke test must pass before
the full split is allowed to run.

## Gate 1 outcome

The corrected full-coverage Cross-Task macro result is Element Accuracy
`28.5204`, Operation F1 `86.9429`, and Step SR `25.9765`, with parse rate
`2080/2080`. All three metrics are within 1 percentage point of the live-paper
Table 4 anchor (`28.3/87.0/25.5`). See [GATE_1_REPORT.md](GATE_1_REPORT.md) and
`artifacts/gate1_audit.json` for the fail-closed count diagnosis and independent
recomputation.

The pipeline is stopped at Gate 1. AndroidControl must not start until this
result receives explicit review approval.
