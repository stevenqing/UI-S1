# AndroidControl Gate 2: OS-Atlas-7B Zero-Shot OOD

Status: `GATE_2_STOP_PREFLIGHT`

Gate 1 was explicitly approved on 2026-07-27. This run is isolated from the
Mind2Web artifacts and is limited to AndroidControl Gate 2.

## Fixed evaluation contract

- Model: released OS-Atlas-7B action model, not OS-Atlas-Base-7B.
- Setting: zero-shot OOD; no AndroidControl supervised fine-tuning.
- Split: AndroidControl test, High setting (high-level instruction only).
- Input identity: official AndroidControl records joined one-to-one to the
  pinned OS-Atlas `eval/data/ac_idx.txt` entries.
- Expected coverage: 7,708 scoreable steps with action history.
- Device policy: exactly one visible GPU.
- Type: exact action-type match.
- Grounding: click/long-press correctness conditional on matching type, using
  Euclidean coordinate distance at most 14% in normalized 0-1000 space.
- TYPE/OPEN_APP: token F1 greater than 0.5; scroll direction and other actions
  require exact argument/action match.
- Primary metrics: Type, Grounding, and step success rate (SR).
- Paper anchor: OS-Atlas Table 5 zero-shot AndroidControl-High OS-Atlas-7B,
  `57.44 / 54.90 / 29.83` percent.

Any mismatch in source identity, split membership, count, prompt, action
history, parser, metric semantics, model revision, or artifact accounting is a
STOP with written diagnosis. Existing workspace-derived JSONL or historical
results cannot substitute for the official source without a verified join.

## Preflight outcome

Gate 2 is stopped before large downloads or inference. The Table 5
OS-Atlas-7B action checkpoint is not publicly available. The public authors'
model list contains Base and Pro checkpoints only; the Pro model card explicitly
states that it is not the model used in Tables 4 and 5 and is trained without
the OOD/SFT experimental constraints. The repository also provides only three
AndroidControl processed examples rather than the full 7,708-row ground truth.

See [GATE_2_PREFLIGHT_REPORT.md](GATE_2_PREFLIGHT_REPORT.md) for evidence and
the exact conditions required to resume.