# AndroidControl Qwen2.5-VL-7B High Lower Bound

Status: `COMPLETE_AUDIT_PASS_STRICT_FORMAT_ZERO`

This controlled base-model lower bound uses the same 7,708 AndroidControl High
identities, prompt contract, vLLM backend, scorer, and audit as the Qwen3 lane.

## Primary result

| Steps | Type Accuracy | Grounding Accuracy | Step SR | Parse rate |
| ---: | ---: | ---: | ---: | ---: |
| 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | **0.0000%** |

No response used the exact lowercase `actions:\n` delimiter required by the
released evaluator. The model emitted `Actions:` in 7,369 rows and singular
`Action:` in 318 rows; 21 rows used neither form. No output repair is applied.

## Diagnostic only

- Flexible parse rate: `95.5890%`.
- Type Accuracy: `66.8656%`.
- Click-only Grounding: `3.7829%`.
- Step SR: `13.7779%`.
- Parser runtime errors: 16.

These values are not primary benchmark results.

## Audit

- Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Coverage: 7,708 ordered unique identities, 0 missing, 0 duplicate, 0 extra.
- Predictions SHA-256:
  `13ac88c79d5dd0504ff8fd537b3c51e028b3e6a317b94c03518ca67a1ae208db`
- Score SHA-256:
  `2b641b8444e337d60d61102c611572ab03db640d2813fbc3ee0f877237c94f00`
- Audit SHA-256:
  `72e8b9d6cea394fb5f9b8040069cce040e797269ba2105d2711a8702eaf9ccf3`