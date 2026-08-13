# Correction 001: Verifier Checkpoint Split

Date: 2026-08-13

Timing: after sequential prereg commit `4831fcf`, before any sequential real-data optimizer step. Real-data optimizer authorization remains false.

The frozen config stated that each verifier holdout used the other three outer-development folds for training but did not specify an independent checkpoint fold for epoch selection. Using the verifier holdout for checkpointing would leak the OOF target; always using the maximum epoch was not preregistered.

The corrected second-layer split reuses the deterministic cyclic contract already used by the cheap ranker. For each outer fold and verifier holdout:

- two development folds fit the verifier;
- the next legal cyclic fold is checkpoint-only;
- the verifier holdout is prediction-only;
- all verifier inputs for fit, checkpoint, and holdout must come from first-layer cheap OOF artifacts for those same rows.

The final verifier epoch is the half-up median of the four selected verifier epochs. This correction changes no data, features, labels, budget grid, threshold grid, or prior result.