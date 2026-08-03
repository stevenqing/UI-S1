# Amendment 006: Variable Official Proposal Superset

Date: 2026-08-01

Status: execution correction after partial raw candidate generation, before merge, scoring, calibration, or any H1 aggregate result.

The official `rank_regions_by_coverage` function deduplicates identical crop regions. Requesting 18 regions therefore returns *up to* 18 unique subimages, not exactly 18. Two observed rows returned 17 and 15 subimages. Both exceed the nine subimages required for N=10.

The fail-closed contract is corrected as follows:

- require at least nine and at most eighteen official subimages per row;
- N=2/4/10 remain exact official ordered prefixes `[0:2]`, `[0:4]`, and `[0:10]`;
- each MDE seed performs the frozen Gumbel reordering over all actually available subimages and selects nine without replacement;
- never duplicate, synthesize, or pad a region;
- record the source candidate-count distribution in the manifest;
- fail closed if any row has fewer than nine subimages.

This correction changes no candidate used by the main N=2/4/10 comparison and does not inspect target boxes or outcomes. Existing completed rows remain valid; failed shards resume from their last persisted row.