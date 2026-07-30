# Preregistration Amendment 004: Independent E5 Merged into W2

Date: 2026-07-30

Status: applied before W2 result generation, following the superseding Collision-Law spec and explicit user instruction to pause E5 and begin Collision-Law work.

## Correction

The initial Collision-Law execution gate incorrectly required the older complementarity E5 to reach `PASS` before W2. The superseding spec states that E5 is canceled as an independent experiment and merged into W2. Requiring the old three-prompt factorial would spend GPU budget on cells that are not part of the Collision-Law perturbation mechanism.

## New gate

W2 may run after:

1. preregistration configs and amendments are committed;
2. W0 regression passes;
3. W1 operator tests pass.

All three conditions are satisfied. W2 view results remain ungenerated at the time of this amendment.

## Reuse of completed E5 work

The paused E5 cell `androidcontrol/original_768` uses the original GUI-R1-7B prompt, greedy decoding, unchanged screenshot, and the 768-token deployment processor profile. It exactly matches W2's GUI-R1-7B / AndroidControl High / `v4` definition and is inherited without rerunning inference.

The inherited cell has:

- four complete shards of 1,927 rows each;
- one merged prediction file of 7,708 rows;
- shard SHA256 values recorded in the complementarity run logs;
- no completed score at pause time.

Scoring is performed by the shared fail-closed Collision-Law scorer, not by resuming the interrupted legacy score process.

## Canceled cells

The remaining old-E5 prompt paraphrase cells are canceled. They are not missing W2 data and must not enter W2's MDE. W2 MDE is computed only across `full`, `v1`, `v2`, `v3`, and `v4` as specified by Collision-Law.