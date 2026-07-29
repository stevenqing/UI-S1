# Mind2Web Qwen2.5-VL Base Lower Bounds

Status: `QWEN_3B_7B_COMPLETE`

These runs use the exact TongUI Mind2Web data, `v2` prompt, two-step `vtvt`
history, local Transformers greedy decoding, and independent episode-macro
scorer. Only the model checkpoint changes.

| Model | Revision | Paper anchor (Element / Op F1 / Step SR) | Status |
| --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | 2.5 / 14.5 / 0.4 | Complete, audit PASS |
| Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 6.2 / 72.8 / 5.0 | Complete, audit PASS |

## Qwen2.5-VL-3B result

All primary values are means of per-episode means over 252 Cross-Task episodes.

| Result | Steps | Element Accuracy | Operation F1 | Step SR | Parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Controlled result | 2,080 | 2.0448% | 14.4735% | 0.7949% | 77.9327% |
| Paper anchor | - | 2.5000% | 14.5000% | 0.4000% | - |
| Delta | - | -0.4552 pp | -0.0265 pp | +0.3949 pp | - |

All three primary metrics are within one percentage point of the paper anchor.
The full merge and strengthened identity/reference/GT-answer/configuration audit
pass.

Format diagnostics:

- Parseable responses: 1,621 / 2,080.
- Strictly supported uppercase actions: 792 / 2,080.
- Lowercase action aliases intentionally not repaired: 790.
- Parsed responses with at least one coordinate above 1: 890.
- Unparsed responses: 459.

## Qwen2.5-VL-7B result

| Result | Steps | Element Accuracy | Operation F1 | Step SR | Parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Controlled result | 2,080 | 7.2573% | 71.6565% | 5.8983% | 88.7019% |
| Paper anchor | - | 6.2000% | 72.8000% | 5.0000% | - |
| Delta | - | +1.0573 pp | -1.1435 pp | +0.8983 pp | - |

Element Accuracy and Operation F1 differ from the paper anchor by slightly
more than one percentage point, so this is retained as a complete controlled
result rather than labeled a strict anchor reproduction. The full merge,
identity/reference/GT-answer/configuration audit, and all 2,080 Qwen3-to-Qwen7
input tensor hash comparisons pass.

Format diagnostics:

- Parseable responses: 1,845 / 2,080.
- Strictly supported uppercase actions: 1,844 / 2,080.
- Parsed responses with at least one coordinate above 1: 260.
- Unparsed responses: 235.
- Other parsed action: one `BACK` action.

## TongUI comparison targets

| Scale | Audited TongUI result (Element / Op F1 / Step SR) | Base result | TongUI - Base |
| --- | --- | --- | --- |
| 3B | 56.2867 / 89.2036 / 51.3824 | 2.0448 / 14.4735 / 0.7949 | +54.2418 / +74.7302 / +50.5874 pp |
| 7B | 60.8320 / 89.1854 / 55.6095 | 7.2573 / 71.6565 / 5.8983 | +53.5747 / +17.5289 / +49.7111 pp |

## Qwen2.5-VL-3B artifacts

- Predictions SHA-256:
	`b3a1f7c742179580f96a8b07953cf985af22c7f10e7592033eeb8de56d039974`
- Score SHA-256:
	`c3a0e8a1b3fdeeedb82412c1883e5d2ee1ba43d9fb96f3647ad37ff42639b64c`
- Audit SHA-256:
	`cc2d093291292d5d37605836e96e62aadbec035cef5a5fd6fdb42c2fad4ac117`

## Qwen2.5-VL-7B artifacts

- Predictions SHA-256:
	`3947772989cfd9e25cf5fadd65ceb772d207c542c0db7b29bacf78b250cc072a`
- Score SHA-256:
	`ce0c5334a4accdcda6559630b050164c015bdcfb23ec49c1e6d584ef0dc1a9c6`
- Audit SHA-256:
	`b263afadc8b0cde917ef26a0a08c3f295d364bac34f50ed9fab68ee48512bb75`