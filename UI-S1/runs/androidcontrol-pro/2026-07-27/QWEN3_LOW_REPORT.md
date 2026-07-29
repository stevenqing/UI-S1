# AndroidControl Qwen2.5-VL-3B Low Lower Bound

Status: `COMPLETE_AUDIT_PASS_STRICT_FORMAT_ZERO`

This controlled base-model lower bound uses the exact 7,708 AndroidControl
identities and Low prompt contract from the audited OS-Atlas-Pro lane. It uses
the same model, vLLM backend, sampling configuration, scorer, and audit as the
Qwen3 High run.

## Primary result

| Steps | Type Accuracy | Grounding Accuracy | Step SR | Parse rate |
| ---: | ---: | ---: | ---: | ---: |
| 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | **0.0000%** |

The model emitted capitalized `Actions:` in 7,606 rows and singular `Action:`
in 102 rows. No row used the exact lowercase `actions:\n` delimiter required
by the released evaluator, so no output repair is applied.

## Diagnostic only

The case-insensitive flexible parser reports:

- Parse rate: `98.6767%`.
- Type Accuracy: `70.0701%`.
- Click-only Grounding: `3.5733%`.
- Step SR: `15.7499%`.
- Parser runtime errors: 264.

Relative to High, the flexible diagnostic changes by `+18.4613 / +1.2526 /
+4.2294 pp` for Type/Grounding/Step SR. These values are not primary benchmark
results.

## Audit

- Coverage: 7,708 ordered unique identities, 0 missing, 0 duplicate, 0 extra.
- Predictions SHA-256:
  `72e9494950afd246f0f284870c281a9cba2c5a0e6ae3cb254686509df44f2f0b`
- Score SHA-256:
  `bd75a3531cc6efc5df6838fc5716afe9d993a84e0070f152f978fc1e743480eb`
- Audit SHA-256:
  `1a7d6c51f5158e0deeb21729010d78f4d1c749ec09ee8adba07a773ed486336f`

The independent audit reconstructs every Low prompt hash and verifies GT
fields, model revision, backend, processor, generation configuration, score
coverage, and final artifact hashes.