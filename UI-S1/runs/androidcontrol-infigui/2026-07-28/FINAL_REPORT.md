# InfiGUI-R1-3B AndroidControl Final Report

Status: `COMPLETE_STRICT_AUDIT_PASS`

## Evaluation contract

- Official InfiGUI AndroidControl release: 1,543 episodes and 8,444 scoreable
  steps per setting.
- Official images, Low/High prompts, `mobile_use` schema, parser, coordinate
  conversion, and evaluator.
- vLLM tensor parallel size 2, temperature 0, seed 42, thinking mode, and up
  to 4,096 new tokens.
- Metrics are `Type Accuracy / Grounding Accuracy / Step Success Rate`.
- This is an independent 8,444-step InfiGUI lane and is not comparable to the
  OS-Atlas strict-parser 7,708-step lane without an explicit evaluator change.

## Final results

| Setting | Reproduced | Paper anchor | Delta (pp) | Result |
| --- | --- | --- | --- | --- |
| Low | **95.97 / 93.87 / 92.09** | 96.0 / 93.2 / 92.1 | -0.03 / +0.67 / -0.01 | **PASS** |
| High | **82.75 / 74.44 / 71.28** | 82.7 / 74.4 / 71.1 | +0.05 / +0.04 / +0.18 | **PASS** |

All six absolute anchor differences are below 1 percentage point.

## Exact counts

| Setting | Rows | Parse success | Type matches | Grounding matches | Grounding denominator | Step successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 8,444 | 8,444 | 8,104 | 4,763 | 5,074 | 7,776 |
| High | 8,444 | 8,444 | 6,987 | 3,777 | 5,074 | 6,019 |

## Independent audit

Both `audit.py --require-complete` runs returned `PASS` with complete coverage,
8,444 unique ordered question IDs, zero mismatches, a clean pinned source
worktree, and evaluator SHA-256
`90ab6a56cae771245282c3ebcaf4cb96016d3c7c05072d155c7db91c5545e1f1`.

| Setting | Prediction artifact SHA-256 | Final audit |
| --- | --- | --- |
| Low | `3f046df81dc586340d1411767968fbf7036e6c5fcffe1f18dd87b88b135e1199` | [final_audit.json](artifacts/low/final_audit.json) |
| High | `07215723ca049f958f4f9e30af47f634d86d4a9df71b00049eecf6b35543f3e8` | [final_audit.json](artifacts/high/final_audit.json) |

Pinned source revision:
`a4fca17809a4395ba1fe08d481bb82c790ea7236`.

## Conclusion

The official InfiGUI-R1-3B AndroidControl Low and High reproductions are
complete and pass the predefined strict audit and paper-anchor gates.