# AndroidControl Pro Low vs High

Both settings use the same OS-Atlas-Pro-7B revision, 7,708 `ac_idx` identities,
images, unified action schema, generation settings, and scorer. The only intended
input difference is whether the current low-level instruction is provided.

| Metric | Low | High | Low - High |
| --- | ---: | ---: | ---: |
| Parse Rate | 100.0000% | 100.0000% | 0.0000 pp |
| Type Accuracy | 93.4743% | 86.3129% | +7.1614 pp |
| Grounding Accuracy | 86.7576% | 77.9466% | +8.8110 pp |
| Step Success Rate | 83.9647% | 71.3285% | +12.6362 pp |

Low includes the current step instruction in addition to the high-level goal
and history. High omits it and therefore requires action planning from the goal
and history. The 12.64-point Step SR gap measures the combined planning and
downstream grounding benefit of the low-level instruction in this offline setup.

Artifacts:

- Low: `artifacts/low_merged/`
- High: `artifacts/high_corrected_merged/`
- Legacy High prompt-whitespace diagnostic: `artifacts/merged/`

These are successor-model results. OS-Atlas-Pro-7B was trained on all seven
agent datasets, so neither row is the paper's zero-shot Table 5 result.