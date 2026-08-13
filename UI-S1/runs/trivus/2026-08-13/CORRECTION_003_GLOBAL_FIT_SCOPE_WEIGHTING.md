# Correction 003: Global Fit-Scope Weighting

Date: 2026-08-13

Timing: after failed replacement nonce `ef9d71ffcf57cc12dc81db8e302196a99003d151424e57d42357437c1b3d3d27`, before any further authorization.

The replacement attempt completed all 60 cheap OOF jobs and produced 120 cheap artifacts. The first eight verifier workers then failed before optimizer construction. Each fold had received TARGET_ONLY weights normalized inside that fold; concatenating two verifier fit folds retained two independently normalized weight vectors. The standardizer correctly rejected these values because the verifier training scope requires one equal-cell normalization over the combined two-fold scope.

Verifier fold loading now preserves unweighted family rows. The two fit folds are concatenated first, then TARGET_ONLY weights and the standardizer are fitted once over the combined scope. Checkpoint and holdout scopes receive their own canonical TARGET_ONLY weights and reuse only the fit-scope standardizer.

The regression test constructs two disjoint pieces containing two Mind2Web cells each. Only after concatenation do all four cells exist; global weighting must then sum to one and pass the TARGET_ONLY standardizer contract.

The failed nonce, receipt, complete cheap artifacts, verifier logs, and attempt remain isolated and unpublished. They cannot be used by a replacement authorization unless a future protocol explicitly binds and validates reuse; the current replacement must rerun both phases.