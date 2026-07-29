# Mind2Web ShowUI-2B Offline Baseline

Status: `COMPLETE_PUBLIC_CHECKPOINT_RESULT`

This run evaluates the public `showlab/ShowUI-2B` checkpoint as the paper's
zero-shot Mind2Web baseline (`ShowUI-ZS`). It must not be reported as the
Mind2Web-fine-tuned `ShowUI` or `ShowUI-dagger` rows because no downstream
Mind2Web state dict is published with this checkpoint.

## Target

- Split: Cross-Task (`test_task`)
- Published ShowUI-ZS anchor: Element Accuracy 21.4%, Operation F1 85.2%,
  Step Success Rate 18.6%
- Public model revision: `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`
- Model weight SHA-256: `68080df785764e98976eb9cc93a07c6c69cf8a6933738496e02aef55b53d2aa3`
- Processor revision: `895c3a49bc3fa70a340399125c650a463535e71c`
- Source revision: `21ed7cb24be0cc877bb8352ee34d58a9aea2c876`
- Prepared metadata SHA-256: `e3dbd288037f14849ca713f92468adaef67f859a3f2f520fffc2b190a82054ef`
- Evaluator: ShowUI's Mind2Web prompt, parser, and episode-macro metrics

## Fail-closed boundary

The source loads an additional downstream state dict only when `version` differs
from `model_id`. This run keeps both values at `showlab/ShowUI-2B`, uses
`--eval_only` and `--lora_r=0`, and labels the result `ShowUI-ZS`. Any result
requiring an unpublished or separately trained Mind2Web state dict belongs in a
different run.

## Result

The four disjoint shards completed with exact 2,080-row coverage. Independent
episode-macro scoring produced 23.3366% Element Accuracy, 81.9835% Operation F1,
and 19.9609% Step Success Rate. See `FINAL_REPORT.md` for anchor deltas,
diagnostics, and artifact hashes.
