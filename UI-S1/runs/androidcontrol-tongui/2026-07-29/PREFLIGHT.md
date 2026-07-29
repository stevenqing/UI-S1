# TongUI AndroidControl Scope Check

Status: `BLOCKED_CHECKPOINT_ACTION_SPACE_MISMATCH`

## Paper Scope

The TongUI paper evaluates ScreenSpot, Mind2Web, AITW, MiniWoB, and Baidu
Experience. It does not report an AndroidControl row. Therefore no TongUI
AndroidControl paper anchor exists to reproduce.

## Public Checkpoint Contract

The official `Bofeee5675/TongUI-3B` and `TongUI-7B` checkpoints are trained on
the AITW mobile schema. Their released prompt supports normalized JSON actions
for click/type/scroll/back/home/enter/status. It does not define the full
AndroidControl action space: LONG_PRESS, OPEN_APP with an app name, and WAIT are
missing.

The repository contains a `mobile_use` tool class with open/wait/long-press
operations, but it is not imported by the official inference or evaluation
paths and is not the checkpoint training target. It cannot be used to claim
checkpoint support for those actions.

## Decision

A partial diagnostic on the action intersection could be run, but it would not
complete AndroidControl and would score unsupported action classes as zero. It
is not a missing paper baseline. A full controlled transfer requires a
checkpoint trained for the complete action schema or an official adapter from
the authors; GT-dependent reinterpretation is prohibited.
