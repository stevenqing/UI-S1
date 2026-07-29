# CogAgent Mind2Web Preflight

Status: `PUBLIC_TRANSFER_COMPLETE_AUDIT_PASS`

## Target

The comparison anchor is `22.4 / 53.0 / 17.6` for
`Element Accuracy / Operation F1 / Step Success Rate` on Mind2Web Cross-Task.

## Verified public release

- Official source: `zai-org/CogVLM` (redirected from `THUDM/CogVLM`).
- GUI-agent checkpoint: `zai-org/cogagent-chat-hf` (redirected from
  `THUDM/cogagent-chat-hf`).
- The checkpoint is an 18B CogAgent model with 1120 x 1120 visual input.
- The official Hugging Face API uses `LlamaTokenizer`,
  `AutoModelForCausalLM(..., trust_remote_code=True)`, and
  `model.build_conversation_input_ids(...)`.
- The official GUI-agent prompt appends `(with grounding)` and emits a plan,
  next action, and a formalized `Grounded Operation`.
- Grounded coordinates are 0-1000 relative bounding boxes in
  `[[x1,y1,x2,y2]]` form.
- The public model card recommends the chat checkpoint for GUI agent and
  grounding tasks and single-round dialogue for each image.

## Evaluator boundary

The public release does not contain a Mind2Web split-specific evaluator,
prompt/history serializer, or action converter. Its generic output is not the
Mind2Web `CLICK`/`TYPE`/`SELECT` JSON consumed by the existing TongUI runner.
In particular:

- a generic `Grounded Operation` must be parsed without consulting the ground
  truth action type;
- the bbox center must be converted from 0-1000 space to normalized
  coordinates before element scoring;
- `TYPE` text can be recovered only when the predicted operation explicitly
  emits it;
- the public operation space has no verified Mind2Web `SELECT` contract;
- the paper anchor cannot be claimed until the exact Cross-Task prompt,
  history, action mapping, and grouping/scoring implementation are recovered.

## Next executable lane

The separately labeled `cogagent-chat-hf public transfer` runner uses the
official single-round `(with grounding)` prompt. Its one-row smoke test and
full 2,080-action run both completed. The final score and audit are in
`artifacts/merged/`; this result is not a paper-anchor reproduction.
