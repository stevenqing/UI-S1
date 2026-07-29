# UI-TARS AndroidControl Public Transfer Report

Status: `FOUR_SETTINGS_COMPLETE_AUDIT_PASS_NOT_PAPER_REPRODUCTION`

## Results

All settings use the ordered 7,708-step OS-Atlas AndroidControl lane with
1,412 episodes. Values are percentages in `Type / Grounding / Step SR` order.

| Model | Setting | Type | Grounding | Step SR | Parse | Paper anchor |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| UI-TARS-2B-SFT | Low | **79.8132** | **80.7730** | **63.5314** | 96.1988 | 98.1 / 87.3 / 89.3 |
| UI-TARS-2B-SFT | High | **69.6549** | **64.4659** | **48.1448** | 96.7566 | 81.2 / 78.4 / 68.9 |
| UI-TARS-7B-SFT | Low | **89.5304** | **85.4152** | **73.8454** | 99.6757 | 98.0 / 89.3 / 90.8 |
| UI-TARS-7B-SFT | High | **79.5797** | **73.4551** | **61.0145** | 99.2735 | 83.7 / 80.5 / 72.5 |

These are complete public-checkpoint controlled transfers, not exact paper
reproductions. The paper does not release its AndroidControl split-specific
prompt serializer or evaluator implementation.

## Comparisons

- 2B Low minus High: Type `+10.1583 pp`, Grounding `+16.3071 pp`, Step SR
  `+15.3866 pp`.
- 7B Low minus High: Type `+9.9507 pp`, Grounding `+11.9601 pp`, Step SR
  `+12.8309 pp`.
- 7B minus 2B on Low: Type `+9.7172 pp`, Grounding `+4.6422 pp`, Step SR
  `+10.3140 pp`.
- 7B minus 2B on High: Type `+9.9248 pp`, Grounding `+8.9892 pp`, Step SR
  `+12.8697 pp`.

## Fixed Contract

- UI-TARS-2B-SFT revision:
  `f366a1db3e7f29635f5b236d6a71dea367a0a700`.
- UI-TARS-7B-SFT revision:
  `3434901a9dd04dd3625617d839a5724fe5e2db20`.
- Low includes the high-level goal, current low-level instruction, and prior
  history; High excludes the current low-level instruction.
- Official generic mobile function-call grammar, extended globally with
  `wait()` because WAIT is part of the benchmark action schema.
- Greedy generation, frequency penalty 1, 128-token cap, seed 0.
- Coordinates are 0-1000 relative points.
- No GT-dependent action, text, or coordinate repair.

## Coverage And Audit

Each setting completed four shards of exactly 1,927 rows. Ordered merge,
identity uniqueness, prompt hashes, model revisions, generation configuration,
score coverage, and artifact hashes passed independently for all four settings.

| Setting | Predictions SHA-256 | Score SHA-256 |
| --- | --- | --- |
| 2B Low | `d862c1c4f7b95231de480e1524aabb87a41f8905076564d7209a196454e5ec94` | `d17b770a885dc38659c2f95914a9fae925738dda02cdd1fcf04b7742d250531a` |
| 2B High | `762a1ef1e20ce4bc39ef9fe51c4ec2169b48ca06455e1ecd9188e2f9a0d6ebd8` | `be9e5ee7ae2608f0c0f78aa146f72d778c43f96e11fe3951443e98f86863e713` |
| 7B Low | `bb24c9ef627324db5dfb9a116a4a45607e7f4835b5da93d5e4459f691355923b` | `0fea2e09b1d7b4200d81eda87adc46be1d2943912eb1d35023f4ba4fadd07cdc` |
| 7B High | `935ae2b79338aad8513aa189675c88db3dfd72ef9b55b9afb3d7422b0f4593e4` | `570a905d0f418683fb22e416d9611797fe2279a885ee2cdd69828e1675f85ab0` |

The shared prepared-data SHA-256 is
`a81c8aa95a773f3d64aa3c51457719664e1992edb23cb68971fe3da06a5e03bb`.
Artifacts are under `artifacts/{2b-low,2b-high,7b-low,7b-high}/merged/`.