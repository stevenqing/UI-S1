# Post-result Correction 002: Missing Cross-Fitted Reliability

Date: 2026-08-11

Timing: identified after first A1 adjudication and before commit, push, or use in a paper claim.

## Defect

Amendment 001 froze cross-fitted candidate reliability as an A1 input. The first implementation used all other frozen public structural features but omitted this scalar. It did not leak labels, but it evaluated a strict subset of the frozen feature state. Therefore its `CLOSE_ROUTING` result is invalid for the registered A1 method.

## Invalid artifacts

The first run is retained at:

`/scratch/workspaceblobstore/care/2026-08-11/INVALID_MISSING_CROSSFITTED_RELIABILITY/`

Its router adjudication SHA-256 is `e60f54ff6a3e6e0aa8a7774bede08d55068e67c0fa5e6fa09a49c34339ff8360`.

## Correction

1. Generate a public metadata sidecar containing only the source key for each of the six shared candidates. It contains no candidate success or target/evaluator field.
2. For each inner checkpoint fit, estimate source success counts only from its three model-training folds.
3. Training-row candidate reliability is Beta(1,1)-smoothed leave-one-row:

$$
r_{si}=\frac{S_s-R_i+1}{N_s-1+2}.
$$

4. Checkpoint-validation rows use the fixed three-fold statistic without their labels.
5. Final outer training estimates statistics on the four development folds and uses leave-one-row values. Outer test uses the fixed four-fold statistic after pretest fsync.
6. Source identity is used only to compute the scalar and is never passed to the model.
7. Architecture, losses, seeds, optimizer, epoch cap, static-arm control, bootstrap, gates, and label-access guard remain unchanged.

This is an implementation correction, not a new feature search. The corrected run supersedes the first A1 result whether its direction changes or not.
