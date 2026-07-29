# AndroidControl Successor Baseline: OS-Atlas-Pro-7B

Status: `LOW_HIGH_COMPLETE_SUCCESSOR_BASELINE`

This run resolves the operational Gate 2 blockage by using the public action
model recommended by an OS-Atlas contributor in GitHub issue #7. It does not
claim to reproduce the unavailable Table 5 checkpoint.

## Contract

- Model: `OS-Copilot/OS-Atlas-Pro-7B` at revision
  `6c0135de0627db98533ac4b47ae71fa17cf21c48`.
- Model status: official released successor action model from paper Section 5.4.
- Data membership: the 7,708 identities in pinned OS-Atlas `ac_idx.txt`.
- Images: community mirror
  `ckg/AndroidControlParsedWithImages-20k-TESTONLY` at revision
  `c33fb032cec29dda202d15dc321c75c31aca9ea3`, accepted only after a complete
  `(episode_id, step_id)` join and annotation cross-check against the local
  independently derived official-source JSONL.
- Settings: AndroidControl Low and High, reported separately.
- Prompt/action schema: pinned OS-Atlas prompt and unified action format.
- Device: exactly one visible GPU.
- Generation: deterministic greedy decoding.
- Evaluation: pinned OS-Atlas AndroidControl evaluator semantics, with an
  independent recomputation from raw model text.

## Interpretation boundary

The Pro model was trained on all seven agent datasets and is not zero-shot OOD
with respect to AndroidControl. Therefore the Table 5 anchor
`57.44/54.90/29.83` is shown only as historical context and must not be used as
a pass/fail target for this run. Results are reported as a new successor
baseline with exact model/data provenance.

The experiment stops before model download if the compact mirror does not cover
all 7,708 `ac_idx` identities or if its images/annotations disagree with the
local official-source derivation.

## Results

AndroidControl-Low completed on all 7,708 steps with an exact released prompt:

- Type Accuracy: `93.4743%`
- Grounding Accuracy: `86.7576%`
- Step Success Rate: `83.9647%`
- Parse Rate: `100%`

The corrected exact-prompt AndroidControl-High run completed on all 7,708 steps:

- Type Accuracy: `86.3129%`
- Grounding Accuracy: `77.9466%`
- Step Success Rate: `71.3285%`
- Parse Rate: `100%`

Low exceeds High by `+7.1614 / +8.8110 / +12.6362` percentage points on
Type/Grounding/Step SR, respectively.

The original High run that omitted one history-terminating newline is retained
under `artifacts/merged` as a legacy diagnostic. The formal High result is under
`artifacts/high_corrected_merged`.

See [LOW_REPORT.md](LOW_REPORT.md), [FINAL_REPORT.md](FINAL_REPORT.md), and
[LOW_HIGH_COMPARISON.md](LOW_HIGH_COMPARISON.md).