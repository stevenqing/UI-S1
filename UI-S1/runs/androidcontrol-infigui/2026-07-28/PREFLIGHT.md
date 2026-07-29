# InfiGUI-R1 AndroidControl Preflight

Status: `LOW_HIGH_COMPLETE_STRICT_AUDIT_PASS`

InfiGUI-R1 is the first remaining candidate with a complete public
AndroidControl release: checkpoint, processed test set, Low/High prompt,
tool-call action schema, coordinate conversion, parser, scorer, and exact run
commands.

## Paper targets

Values are `Type / Grounding / Step SR`.

| Setting | InfiGUI-R1-3B anchor |
| --- | --- |
| Low | `96.0 / 93.2 / 92.1` |
| High | `82.7 / 74.4 / 71.1` |

## Pinned release

- Source: `Reallm-Labs/InfiGUI-R1`
  - revision: `a4fca17809a4395ba1fe08d481bb82c790ea7236`
- Model: `Reallm-Labs/InfiGUI-R1-3B` / `InfiX-ai/InfiGUI-R1-3B`
  - both names resolve to revision
    `7b0e1de35afb807c6bfa70a2b85df24cf298e73d`
  - model shard SHA-256:
    - `d22bd8182d71bd60728f5c4c48b97e648f86df66d9ef65723a86cf6d1e4a316e`
    - `a96034dd1922edce8337a1fa39a197a9dfe33ae92b47721a2c39c09477eb50af`
- Dataset: `Reallm-Labs/android_control_test`
  - revision: `92bf0d54e371474bff2a94dd93f087ec6940b54d`
  - archive size: 6,429,419,293 bytes
  - archive SHA-256:
    `731037cf4f7e9a9f75ddf8cb58b4041e24118b785aaa4651d6d479d066d0f061`
  - license: Apache-2.0

## Released contract

- Low prompt includes high-level goal, current step query, and action history.
- High prompt includes high-level goal and action history.
- Thinking mode is required by the published reproduction command.
- Output is a `mobile_use` tool call with actions covering `click`, `type`,
  `swipe`, `long_press`, `open`, `wait`, `system_button`, and `terminate`.
- Coordinates are absolute in the smart-resized image and normalized by the
  released evaluator before matching.
- Official generation uses vLLM, temperature 0, seed 42, and up to 4,096 new
  tokens.

## Metric boundary

The released InfiGUI evaluator differs from the OS-Atlas unified evaluator. It
uses candidate bounding boxes plus a 4% screen-distance fallback for clicks,
allows an `open` target to match a click on an app icon, and evaluates its own
tool-call schema. Results must therefore be reported as an official InfiGUI
reproduction lane and must not be mixed with OS-Atlas strict-parser numbers.

## Final gates

Completed artifact checks:

- Both model shard sizes and SHA-256 values match the pinned release.
- Dataset archive size and SHA-256 match the pinned release.
- Archive contains no absolute or path-traversal members.
- Extracted data contains 1,543 episodes, 8,444 unique scoreable steps, 9,987
  PNG files, 0 missing images, and 0 malformed episodes.
- The pinned OS-Atlas 7,708 identities are an exact subset of the InfiGUI
  split; InfiGUI has 736 additional steps and no missing OS-Atlas identity.
- All 7,708 overlapping action types agree after schema translation.
- Overlapping PNG bytes differ, so official InfiGUI images must be used.

Completed runtime gates:

- High/Low one-job smoke tests pass the full official prompt, tool-call parser,
  coordinate conversion, and per-step scorer path.
- High and Low full lanes completed with tensor parallel size 2 on GPUs 0-1
  and 2-3, respectively, with exactly 8,444 rows per setting.
- Persisted rows passed ordered question ID, GT field, prompt hash, parser, and
  independently recomputed score checks.
- Standalone `audit.py` rebuilds all 8,444 reference jobs without loading the
  model, re-parses raw responses, recomputes official scores, and verifies all
  row provenance. It withholds metrics for partial coverage and supports a
  `--require-complete` final gate.
- Pinned evaluator SHA-256:
  `90ab6a56cae771245282c3ebcaf4cb96016d3c7c05072d155c7db91c5545e1f1`.
- High and Low final `--require-complete` audits pass with zero mismatches; the
  pinned source revision and worktree checks also pass.

Final results (`Type / Grounding / Step SR`):

| Setting | Reproduced | Paper anchor | Delta (pp) |
| --- | --- | --- | --- |
| Low | `95.97 / 93.87 / 92.09` | `96.0 / 93.2 / 92.1` | `-0.03 / +0.67 / -0.01` |
| High | `82.75 / 74.44 / 71.28` | `82.7 / 74.4 / 71.1` | `+0.05 / +0.04 / +0.18` |

All gates pass and both settings are reportable. See
[FINAL_REPORT.md](FINAL_REPORT.md) for exact counts, artifact hashes, and final
audit links.