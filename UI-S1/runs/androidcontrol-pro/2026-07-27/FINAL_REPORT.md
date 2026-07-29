# AndroidControl OS-Atlas-Pro-7B Successor Baseline

Verdict: **COMPLETE**

This experiment resolves the runnable-baseline blockage with the official
public OS-Atlas action model recommended by an OS-Atlas contributor. It does not
claim to reproduce the unavailable Table 5 zero-shot checkpoint.

Byte-level comparison against the released multi-step OS-Atlas samples found
that the first High run used one newline after non-empty history where the
released prompt uses two. High was rerun with the corrected exact prompt. The
legacy raw artifacts remain under `artifacts/merged`; all formal High numbers
below use `artifacts/high_corrected_merged`.

## Results

| Metric | Result | Count |
| --- | ---: | ---: |
| Upstream-exact parse rate | 100.0000% | 7,708 / 7,708 |
| Type Accuracy | 86.3129% | 6,653 / 7,708 |
| Grounding Accuracy | 77.9466% | click-correct / click-type-match |
| Coordinate Grounding incl. long press | 77.9574% | coordinate-correct / coordinate-type-match |
| Step Success Rate | 71.3285% | 5,498 / 7,708 |

The flexible-parser diagnostic produces exactly the same metrics as the pinned
upstream lowercase `actions:` parser. No score depends on added parser leniency,
and scoring recorded zero runtime errors.

Per-action successful steps:

| Action | Success / Total |
| --- | ---: |
| CLICK | 3,181 / 4,598 |
| SCROLL | 852 / 1,138 |
| OPEN_APP | 445 / 554 |
| TYPE | 430 / 569 |
| WAIT | 381 / 527 |
| PRESS_BACK | 207 / 315 |
| LONG_PRESS | 2 / 7 |

## Data Audit

- OS-Atlas `ac_idx.txt`: 7,708 identities, SHA-256
  `4a0008a58b82495a7c77745eca167f13c2e4fe683cd865490c13d249ddfb83f8`.
- Source test mirror: `aliaagheis/android-control` revision
  `0519923f5e8882c679a4ec8a4ccc91b93d79e8ae`.
- Mirror: 30 parquet files, 1,543 unique episodes, 8,444 original actions.
- Join: all 7,708 requested identities found; 0 missing and 0 duplicate.
- Extracted images: 7,708 valid PNGs, 5,135,648,959 bytes; 0 corrupt or
  dimension mismatches.
- Goals and action identities match the independently derived local copy for
  all 8,444 actions. The mirror additionally preserves official scroll
  direction values absent from that older derived copy.
- The first three generated GT actions exactly match the pinned OS-Atlas sample:
  `OPEN_APP [PocketBook]`, `LONG_PRESS [[108,272]]`, and `CLICK [[950,87]]`.

An earlier candidate mirror was rejected before model inference because only
579 of its 8,238 identities overlapped `ac_idx`. Its data remains under
`data/mirror` as a documented rejected source and is not used in results.

## Model And Execution

- Model: `OS-Copilot/OS-Atlas-Pro-7B`.
- Revision: `6c0135de0627db98533ac4b47ae71fa17cf21c48`.
- Model directory: 52 files, 16,594,385,553 bytes.
- All four safetensor shards were checked against their pinned SHA-256 values.
- Prompt: pinned OS-Atlas unified action prompt; AndroidControl-High includes
  high-level goal and previous low-level actions but omits the current
  low-level instruction.
- Generation: model-card settings (`temperature=0.01`, `top_k=1`,
  `top_p=0.001`), 128-token cap, slow processor preserved explicitly by the
  downloaded model configuration.
- Hardware: four NVIDIA A100-SXM4-80GB GPUs, one model process per visible GPU.
- Batch size: 4. Batch 4 was verified byte-for-byte against batch 1. Batch 8
  was rejected after OOM and was not used.
- Corrected High used four disjoint 1,927-row shards, merged back in exact
  `ac_idx` order.

## Reproducibility

- Raw predictions: `artifacts/high_corrected_merged/predictions.jsonl`, SHA-256
  `f8dbeeb012a1ffb4183a78809b35db75e3a63e4ef36a4f6dc9a1f47ec055a9aa`.
- Score: `artifacts/high_corrected_merged/score.json`, SHA-256
  `f1f61f142d6bd6aa95afd7a467b804dfc69f71e24f121210775d6cca085c0ee0`.
- Prepared input: `data/prepared/ac_high.jsonl`, SHA-256
  `a81c8aa95a773f3d64aa3c51457719664e1992edb23cb68971fe3da06a5e03bb`.
- Environment: Python 3.10.20, uv 0.11.32, PyTorch 2.4.1+cu121,
  Transformers 4.49.0, qwen-vl-utils 0.0.14.
- Lock file: `uv.lock`.
- Harnesses: `prepare_data.py`, `infer.py`, `merge_predictions.py`, and
  `score.py`.

## Interpretation Boundary

The original Gate 2 target was OS-Atlas-7B zero-shot OOD on
AndroidControl-High (`57.44 / 54.90 / 29.83` in Table 5). That exact action
checkpoint is not public. The authors instead released and recommended
OS-Atlas-Pro-7B, whose model card states it is the Section 5.4 model trained on
all seven agent datasets. Consequently, the `86.31 / 77.95 / 71.33` result is a
successor baseline and must not be presented as a reproduction or improvement
over the Table 5 zero-shot row.

The original fail-closed diagnosis remains at
`runs/androidcontrol/2026-07-27/GATE_2_PREFLIGHT_REPORT.md`.