# Correction 004: prospective holdout labels were opened

Date: 2026-08-14

Status: `CORRECTED_BEFORE_EXPLORATORY_METHOD_SPEC`

The screen seal correctly excluded holdout screens from Q1, Q2, Q3, Q4, and the shared-target aggregates. However, the implementation did not preserve the stronger promise that holdout labels would never be read:

- `prepare_private_manifest.py` parsed every row in all five private-label files to validate schema, candidate-array width, and identity.
- `q3_q4.py` loaded all private-label rows and all reference rows into memory before indexing only the exploratory Q2 keys.

Therefore the 30% screen subset is statistically excluded from the reported XSCR quantities but is not an unread prospective internal holdout. It cannot support independent or confirmatory evaluation in a later method round. XSCR remains post-selection descriptive feasibility, and its Q1-Q4 numbers are unchanged.

Any exploratory follow-up on the current data must use explicitly post-selection nested evaluation over already-opened labels. Independent validation requires new data whose labels have not been accessed.