# AgentTrek Mind2Web Preflight

Status: `BLOCKED_UNPUBLISHED_VISION_CHECKPOINT`

## Paper Target

AgentTrek Table 7 reports the Multimodal-Mind2Web Cross-Task result
`45.5 / 84.9 / 40.9` for `Element Accuracy / Operation F1 / Step Success
Rate`.

The row is explicitly:

- observation: `Image`;
- base architecture: `Qwen2-VL`;
- training method: `+ AT` (AgentTrek visual trajectory data only);
- inference: pure screenshot input with coordinate generation.

This is not the public text-based AgentTrek-32B model.

## Public Release Boundary

The official repository publishes only `xlangai/AgentTrek-1.0-32B`, a
Qwen2.5-32B-Instruct text web agent used with BrowserGym/WebArena. The release
README marks AgentTrek-7B and AgentTrek-72B as unreleased and contains no
Multimodal-Mind2Web evaluator or vision checkpoint.

The paper's visual model is a Qwen2-VL model fine-tuned on 10,000 selected
AgentTrek visual trajectories. No model ID, fixed revision, state dict, prompt,
or Mind2Web conversion/evaluator for this visual model is present in the public
release.

## Why The Public 32B Model Is Not A Substitute

- Architecture differs: text Qwen2.5-32B versus multimodal Qwen2-VL.
- Observation differs: accessibility tree/text versus screenshot pixels.
- Action interface differs: BrowserGym/Playwright actions versus generated
  visual coordinates.
- Evaluation differs: online WebArena/MiniWoB versus offline Multimodal-
  Mind2Web.

Running the public 32B model on the 2,080 screenshot rows would not reproduce
or meaningfully transfer the paper row because it cannot consume the required
visual observation contract.

## Resume Condition

Execution can resume only when the authors publish the Qwen2-VL `+ AT` visual
checkpoint and enough of the original Mind2Web prompt/history/action converter
to create a GT-independent parser. Until then the row remains blocked; no
fabricated or architecture-substituted score is reportable.

## Sources

- Paper: `arXiv:2412.09605v2`, Table 7 and Appendix J.2.
- Official repository: `xlang-ai/AgentTrek`.
- Public text checkpoint: `xlangai/AgentTrek-1.0-32B`.
