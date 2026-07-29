# Mind2Web TongUI-7B Cross-Task Report

Status: `COMPLETE_CONTROLLED_RESULT`

## Scope

- Model: `Bofeee5675/TongUI-7B`
- Revision: `a3e0cf46c3164bbd885dea2694f2ad7a31f1661d`
- Split: Cross-Task, 2,080 scoreable steps in 252 episodes.
- Prompt: released `v2`, two-step `vtvt` visual/thought/action history.
- Inference: local Transformers greedy decoding, FlashAttention 2, bf16.
- Training scope includes Mind2Web; this is not a zero-shot result.

## Published Anchor

| Model | Element Accuracy | Operation F1 | Step Success Rate |
| --- | ---: | ---: | ---: |
| TongUI-7B (GUI-Net-1M) | 58.1% | 88.7% | 53.4% |

## Controlled Result

All primary values are means of per-episode means over 252 Cross-Task episodes.

| Result | Steps | Element Accuracy | Operation F1 | Step SR | Parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public checkpoint | 2,080 | 60.8320% | 89.1854% | 55.6095% | 99.9519% |
| Paper TongUI-7B | - | 58.1000% | 88.7000% | 53.4000% | - |
| Delta | - | +2.7320 pp | +0.4854 pp | +2.2095 pp | - |

This is a complete controlled result for the pinned public checkpoint. It is
not labeled an exact paper reproduction because the Element and Step metrics
differ from the paper anchor by more than one percentage point.

One response (index 1332) emitted an invalid position `[_from]` and is scored as
an unparsed failure without repair. All other 2,079 responses parse and use a
supported Mind2Web action.

Diagnostics:

- Parsed actions: `CLICK` 1,779, `TYPE` 248, `SELECT` 52.
- Micro Element / Operation F1 / Step SR:
	57.6923% / 88.5121% / 52.9327%.

## Evaluation Boundary

The released vLLM evaluator is excluded because it uses ground-truth-dependent
action repair and best-of-N selection. This run uses one greedy response per
step without repair.

## Artifacts

- Predictions SHA-256:
	`89c4630cbef35f5cf48345c8908d173d5ee8fb91d41f67b0121eb919d16b5b77`
- Score SHA-256:
	`4ae8b35d0a075c639620ec48b0d1a6e04a17e780d812237f30472caa12ecae24`
- Audit SHA-256:
	`8df59319d377995ffe455860daaf75caa50fdf8548a8818f11bbf1b1a4b2fd5a`
- Metadata SHA-256:
	`6e86b61ab6b8c657cabadc73de9df1f844dc39e4904228c8b2b5a18b68640d2d`
- Audit: `PASS`, 2,080 unique identities, 252 episodes.
