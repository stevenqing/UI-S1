# AndroidControl Unreleased Baselines Preflight

Status: `GUI_R1_UI_R1_UI_R1_E_UI_AGILE_BLOCKED`

## GUI-R1

The InfiGUI-R1 comparison table reports AndroidControl-High:

- GUI-R1-3B: Type 58.0, Grounding 56.2, Step SR 46.6.
- GUI-R1-7B: Type 71.6, Grounding 65.6, Step SR 51.7.

No official public checkpoint, fixed model revision, AndroidControl prompt,
action converter, or evaluator was located. The comparison numbers cannot be
reproduced from the InfiGUI-R1 checkpoint because it is a different model and
uses its own official 8,444-step lane.

## UI-R1

The same comparison table reports AndroidControl-Low Type 94.3, Grounding 82.6,
and Step SR 88.5. No official checkpoint or complete evaluation contract was
located. `UI-R1` is not an alias for `InfiGUI-R1`; substituting the released
InfiGUI model would fabricate the baseline identity.

## UI-R1-E And UI-AGILE

No unambiguous paper/project definition, official model ID, checkpoint revision,
AndroidControl anchor, or evaluator contract was located for either name. They
cannot be scheduled from a name alone.

## Resume Conditions

Each row requires all of:

1. an official project or paper identity;
2. a public checkpoint at a fixed revision corresponding to the reported row;
3. the AndroidControl split/sample count and Low/High setting;
4. prompt, action grammar, coordinate space, and scoring implementation.

Until those are available, these rows remain blocked rather than incomplete GPU
runs.
