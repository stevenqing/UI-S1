# Mind2Web TongUI-3B Cross-Task Report

Status: `COMPLETE_CONTROLLED_RESULT`

## Scope

- Model: `Bofeee5675/TongUI-3B`
- Revision: `8c08f9d982bd8bd00b31aee11d8a3a1fa3498836`
- Training scope: GUI-Net-1M plus released downstream data recipe including
  Mind2Web; this is not a zero-shot Mind2Web result.
- Split: Cross-Task, 2,080 scoreable steps in 252 episodes.
- Prompt: released `v2`, two-step `vtvt` visual/thought/action history.
- Inference: local Transformers greedy decoding, FlashAttention 2, bf16.

## Published Anchor

| Model | Element Accuracy | Operation F1 | Step Success Rate |
| --- | ---: | ---: | ---: |
| TongUI-3B (GUI-Net-1M) | 53.4% | 89.0% | 48.8% |

## Controlled Result

All primary values are means of per-episode means over 252 Cross-Task episodes.

| Result | Steps | Element Accuracy | Operation F1 | Step SR | Parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public checkpoint | 2,080 | 56.2867% | 89.2036% | 51.3824% | 100.0000% |
| Paper TongUI-3B | - | 53.4000% | 89.0000% | 48.8000% | - |
| Delta | - | +2.8867 pp | +0.2036 pp | +2.5824 pp | - |

This is a complete controlled result for the pinned public checkpoint. It is
not labeled an exact paper reproduction because the Element and Step metrics
differ from the paper anchor by more than one percentage point.

Diagnostics:

- Supported action outputs: 2,080 / 2,080.
- Parsed actions: `CLICK` 1,791, `TYPE` 245, `SELECT` 44.
- Micro Element / Operation F1 / Step SR:
  53.6058% / 88.9183% / 48.9904%.

## Evaluation Boundary

The released vLLM evaluator is excluded because it samples multiple responses,
selects against ground truth, and repairs mismatched action types with the
ground-truth action. This run does not use action repair, best-of-N selection,
or parser leniency.

## Artifacts

- Predictions SHA-256:
  `688d6ce969d8f8b7af5fa46f78577c0b0c3ad98bb9d679998ffd5c8a59e88734`
- Score SHA-256:
  `0eadab49900e25803d05ed46f4a346f491472673720c2f8c2c9359f1b9867a51`
- Audit SHA-256:
  `6809df7b8b29a9fcde629f6a2b1224878c072927b62b03428ccfd5bdff890fdc`
- Metadata SHA-256:
  `6e86b61ab6b8c657cabadc73de9df1f844dc39e4904228c8b2b5a18b68640d2d`
- Audit: `PASS`, 2,080 unique identities, 252 episodes.