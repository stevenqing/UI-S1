# Gate 2 Preflight Report: AndroidControl

Verdict: **STOP**

Gate 1 approval was received and Gate 2 preflight was started. No large dataset
or model download and no AndroidControl inference was started because the exact
paper-comparable model and complete processed evaluation release are unavailable.

## Intended Contract

The target is the OS-Atlas paper Table 5 zero-shot OOD row for
AndroidControl-High: 7,708 test steps, high-level instruction plus action
history, with Type/Grounding/SR anchor `57.44/54.90/29.83` percent. Click and
long-press coordinates use the paper's 14%-of-screen normalized-distance rule.

## Blocking Findings

### 1. Table 5 checkpoint is not public

`OS-Copilot/OS-Atlas-7B` returns HTTP 401 from the Hugging Face model API. The
public author listing exposes `OS-Atlas-Base-7B`, `OS-Atlas-Base-4B`,
`OS-Atlas-Pro-7B`, and `OS-Atlas-Pro-4B`, but not the action checkpoint used for
Tables 4 and 5.

Neither public alternative is valid:

- Base-7B is a GUI grounding model, not the action model evaluated in Table 5.
- Pro-7B is the Section 5.4 all-dataset model. Its own model card explicitly
  states that it differs from Tables 4 and 5 and is not constrained to the OOD
  and SFT experimental conditions. It therefore leaks the target task family
  relative to the requested zero-shot OOD row.

### 2. Full processed test release is absent

The pinned OS-Atlas repository is commit
`bad08407ab54b5bf6c17a69fe1ced476b9494926`. Its `ac_idx.txt` has exactly 7,708
lines (SHA-256
`4a0008a58b82495a7c77745eca167f13c2e4fe683cd865490c13d249ddfb83f8`),
but `ac_test.jsonl` and `ac_low_test.jsonl` contain only three sample rows each.
The repository does not include the complete conversion or inference pipeline.

This matters because the full High prompt, low-level-instruction omission,
history serialization, coordinate normalization, generated-text extraction,
and target conversion must be held constant to compare against Table 5.

### 3. Source images are not present

The local derived AndroidControl JSONL has 1,543 episodes and 8,444 steps. It
contains all 7,708 `ac_idx` identities plus 736 excluded steps, so it is useful
as a join cross-check. However, none of its 8,444 referenced screenshots exists.

The official Google bucket is accessible and contains 20 gzip TFRecord shards
totalling 49,930,232,975 bytes, plus `splits.json` and
`test_subsplits.json`. Disk space is sufficient, but downloading 50 GB cannot
repair the missing model or missing OS-Atlas conversion contract.

## Why No Substitute Run Was Started

Running Pro-7B on the local 1,543-episode task-success harness would produce a
number, but it would change the checkpoint, dataset membership, granularity,
prompt/history, metric semantics, and anchor. Existing workspace reports around
8-9% use that different task-success harness and are not evidence for the paper's
29.83% step SR.

Under the fail-closed policy, such a run must not be labeled Gate 2 reproduction.

## Resume Conditions

1. Supply the exact OS-Atlas-7B Table 5 action checkpoint with an immutable
   revision or checksum.
2. Supply the complete 7,708-row AndroidControl-High processed evaluation file,
   or the exact OS-Atlas conversion and inference code used for Table 5.
3. Then download and checksum the official 20 shards and split files, recreate
   screenshots, prove a one-to-one ordered `ac_idx` join, run a one-step smoke
   test on one visible GPU, and only then run the full split.

Gate 2 remains stopped at preflight. Gate 1 artifacts are unchanged.