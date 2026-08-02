# Amendment 004: Third Drop-In Rule

Date: 2026-08-02

Status: frozen after F1/F4 and before computing this additional zero-GPU comparison. It resolves the Closing requirement for a third drop-in case; it does not alter R1, F1's six preregistered comparisons, F2, F3, or F4.

Use the unchanged `mvp_graph_centroid` implementation from CCM H1 `aggregators_coord.py`. For each N12 row, apply it to the exact ordered V-only or Mixed candidate points and candidate metadata. No threshold, coverage, grouping, centroid, or tie behavior changes.

Report V-only and Mixed accuracy, paired delta, 99% fold-stratified application bootstrap CI, and one-sided plus-one p-value with the same seed/resamples as F1. Add the result under `third_drop_in_graph_centroid` in `f1_paired_bootstrap.json`.

This becomes R2's third deployable rule example. pass@12 remains an oracle/headroom diagnostic and is not called a drop-in rule.
