# AndroidControl Qwen2.5-VL-7B Low Lower Bound

Status: `COMPLETE_AUDIT_PASS_STRICT_FORMAT_ZERO`

This controlled base-model lower bound uses the exact 7,708 AndroidControl Low
identities and the same vLLM, sampling, scorer, and audit contracts as Qwen7
High.

## Primary result

| Steps | Type Accuracy | Grounding Accuracy | Step SR | Parse rate |
| ---: | ---: | ---: | ---: | ---: |
| 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | **0.0000%** |

No response used the exact lowercase `actions:\n` delimiter. The model emitted
`Actions:` in 7,675 rows and singular `Action:` in 29 rows; four rows used
neither form. No output repair is applied.

## Diagnostic only

- Flexible parse rate: `99.5459%`.
- Type Accuracy: `78.5807%`.
- Click-only Grounding: `5.3366%`.
- Step SR: `17.9165%`.
- Parser runtime errors: 8.

Relative to High, Low improves the flexible diagnostic by `+11.7151 / +1.5537
/ +4.1386 pp` for Type/Grounding/Step SR. These values are not primary
benchmark results.

## Audit

- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Coverage: 7,708 ordered unique identities, 0 missing, 0 duplicate, 0 extra.
- Predictions SHA-256:
  `9dab709340a6463c505382bf263c6e88a1ba5a24bd2b9fcf32eddb84538bb26b`
- Score SHA-256:
  `f21ed81a0eb777a8bf4044a15dd33b49d17e9ad361e9f9c60eb49490f5b206d7`
- Audit SHA-256:
  `1edf9c345368a7d8f15938d3e4df476eca6712b6ee24a2aa45f9f373273b9ffd`

Shard 0 used a fixed 8 GiB KV cache to bypass shared-GPU memory-profiling
instability. This changes cache capacity only; prompt, weights, sampling, and
outputs remain under the same audited contract.