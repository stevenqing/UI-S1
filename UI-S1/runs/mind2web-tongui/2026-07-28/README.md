# Mind2Web TongUI Offline Baselines

Status: `TONGUI_3B_7B_QWEN_3B_7B_COMPLETE`

This run targets the public GUI-Net-1M TongUI checkpoints on Mind2Web
Cross-Task. TongUI-3B is the first execution target; TongUI-7B follows through
the same validated harness.

## Published anchors

| Model | Element Accuracy | Operation F1 | Step Success Rate |
| --- | ---: | ---: | ---: |
| TongUI-3B (1M) | 53.4% | 89.0% | 48.8% |
| TongUI-7B (1M) | 58.1% | 88.7% | 53.4% |

## Pinned inputs

- Source revision: `82631b583180cdc870b4774d2495dd94f1558c46`
- TongUI-3B revision: `8c08f9d982bd8bd00b31aee11d8a3a1fa3498836`
- TongUI-7B revision: `a3e0cf46c3164bbd885dea2694f2ad7a31f1661d`
- Benchmark metadata revision: `b212dcd803bd1be8318b9cfbe4912f385a748ff7`
- Split: Cross-Task (`hf_test_task_with_thoughts`)

## Fail-closed boundary

The released vLLM evaluator is not admissible: it samples three responses,
selects the best against ground truth, and overwrites a mismatched predicted
action with the ground-truth action. This run uses only the local Transformers
greedy generation path and independently scores raw responses without action
repair or best-of-N selection.

The current public checkpoints are trained on GUI-Net-1M and the released
training configuration includes Mind2Web training data. Results must not be
described as zero-shot Mind2Web evaluation.

## Current execution state

TongUI-3B completed four 520-row shards, exact ordered merge, independent
scoring, and identity/configuration audit. Its episode-macro Element Accuracy,
Operation F1, and Step SR are 56.2867%, 89.2036%, and 51.3824%. The audit passes
with 2,080 unique identities and 252 episodes. See `TONGUI_3B_REPORT.md`.

TongUI-7B revision `a3e0cf46c3164bbd885dea2694f2ad7a31f1661d` is downloaded
and hash-verified. It completed four 520-row shards, ordered merge, independent
scoring, and audit. Its episode-macro Element Accuracy, Operation F1, and Step
SR are 60.8320%, 89.1854%, and 55.6095%. See `TONGUI_7B_REPORT.md`.

The Qwen2.5-VL-3B and Qwen2.5-VL-7B base checkpoints are also downloaded at
revisions `66285546d2b821cf421d4f5eb2576359d3770cd3` and
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Their shard hashes and model
indices pass validation; inference has not started.

Qwen2.5-VL-3B completed all four shards and audit. Its episode-macro Element
Accuracy, Operation F1, and Step SR are 2.0448%, 14.4735%, and 0.7949%; all are
within one percentage point of the paper lower-bound anchor. Qwen2.5-VL-7B
completed with 7.2573% Element Accuracy, 71.6565% Operation F1, and 5.8983%
Step SR. Its full audit passes, but two anchor deltas are slightly above one
percentage point, so it is a complete controlled result rather than a strict
anchor reproduction.
