# AndroidControl Qwen2.5-VL-3B High Lower Bound

Status: `COMPLETE_AUDIT_PASS_STRICT_FORMAT_ZERO`

This is a controlled base-model lower bound, not an OS-Atlas Table 5
reproduction. It uses the exact 7,708 AndroidControl identities and High prompt
contract from the audited OS-Atlas-Pro lane, with no output repair.

## Contract

- Model: `Qwen/Qwen2.5-VL-3B-Instruct` revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Backend: vLLM 0.11.0, four modulo shards, batch size 32.
- Environment: torch 2.8.0, Transformers 4.57.1, explicit slow processor.
- Generation: 128 tokens, temperature 0.01, top-k 1, top-p 0.001, seed 0.
- Scoring: upstream OS-Atlas exact parser is primary; flexible parsing is
  diagnostic only.

## Primary result

| Steps | Type Accuracy | Grounding Accuracy | Step SR | Parse rate |
| ---: | ---: | ---: | ---: | ---: |
| 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | **0.0000%** |

The base model never emitted the exact lowercase `actions:\n` delimiter
required by the released evaluator. It emitted capitalized `Actions:` in 7,294
rows and singular `Action:` in 414 rows. Lowercasing or otherwise repairing the
model output would change the evaluator contract and is therefore not applied.

## Diagnostic only

The case-insensitive flexible parser reports:

- Parse rate: `94.6290%`.
- Type Accuracy: `51.6087%`.
- Click-only Grounding: `2.3207%`.
- Step SR: `11.5205%`.
- Parser runtime errors: 263.

These values are not primary benchmark results.

## Audit

- Coverage: 7,708 ordered unique identities, 0 missing, 0 duplicate, 0 extra.
- Predictions SHA-256:
  `50cba7b2e50a101a6e2d7c72fd4ac110aabab869ed0a58fd0eeb1dbe4b99af8e`
- Score SHA-256:
  `ddd334276a5460cd980aa3d12dd16f9f9ba4a13dbf0f9f3023cc710974792dda`
- Audit SHA-256:
  `6be903dc2c2805fe463dccaced61bc5c97c4e9cbf398a3c8d53be900d83ab479`

The independent audit reconstructs every prompt hash and verifies GT fields,
model revision, backend, processor, generation configuration, score coverage,
and final artifact hashes.