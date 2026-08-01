# Amendment 001: Shared View Extension

Date: 2026-08-01

Status: frozen after preregistration commit `00aa688` and one unscored schema smoke example per extended model, before production candidate generation or any L1/L2 measurement.

Existing H3 artifacts provide views 0-3 for Qwen3-VL-8B and UI-TARS-7B under shared GTA1 proposal geometry. L1 N=24 requires views 0-7 for each model; L2 fixed-budget pools require views 0-11 for single/two-lineage constructions.

Generate only missing views 4-11 for Qwen3 and UI-TARS. Regions are exact GTA1 official ordered regions from each row's H1 superset. Do not rerun or replace views 0-3. The generator records original view indices and region hashes. Merge requires:

- 1,581 identities per model;
- unique view indices 0-11 per identity;
- exact region equality with GTA1 for each view;
- no target bbox access during inference;
- no duplicated or padded region.

The frozen N12 manifest SHA-256 is `2a7233cf0fbab109ea481ba891d2d7e65a481c48cd73c623d7d76fc61c06ae17`. Its 1,581 sorted identities and stable indices exactly match the existing H3 N4 manifest (SHA-256 `8996d33b92dc1a3c69700905277fa7e5f023e91c9d8364fb26b0bd2dd4dbfc49`), and every H3 N4 region list equals the corresponding N12 prefix. Existing views 0-3 retain the N4 candidate hash; new views 4-11 retain the N12 candidate hash. Merge validation compares each view's region to the N12 manifest rather than requiring these two prefix hashes to be equal.

Rows with fewer than 12 GTA1 candidates fail closed. H1 audit already confirms all rows have at least 16 total candidates, so the extension is available for the full dataset.

V-only N=24 remains unavailable because at least one row has only 16 total unique GTA1 candidates. Mixed N=24 remains available because it needs only views 0-7 from each model.