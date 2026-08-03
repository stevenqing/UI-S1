# Amendment 001: F1/F4 Operations

Date: 2026-08-02

Status: frozen after Closing preregistration and before any F1/F4 result.

## F1

Use the application-to-fold map returned by the frozen Allocation-Law `group_folds` implementation. For each bootstrap replicate, independently resample with replacement the original number of application groups within each of the five frozen folds. Pool all rows belonging to sampled groups, preserving group multiplicity.

For paired binary outcomes `left` and `right`, compute each replicate's row-weighted mean of `left - right`. The 99% interval is the 0.5th and 99.5th percentile. The one-sided p-value is `(1 + number of replicates with delta <= 0) / (10000 + 1)`. The point estimate is the unresampled paired row mean.

Comparisons are exactly:

- mixed N12 M1 versus V-only/GTA1 N12 M1;
- mixed N12 B3 versus V-only/GTA1 N12 B3;
- mixed N12 M1 versus Qwen3 N12 M1;
- mixed N12 M1 versus UI-TARS N12 M1;
- mixed N16 M1 versus V-only N16 M1;
- mixed N16 B3 versus V-only N16 B3.

The GTA1 alias is reported once with both names. No paper-only baseline enters bootstrap. MDE is `0.007043345177520599`; report point delta divided by MDE without using it as a significance threshold.

## F4

Reconstruct the exact frozen V-only N12 and Mixed N12 candidate pools and row-level pass@12. Compute normalized target area as bbox area divided by image area. Sort by `(area_ratio, row_id)` and split with NumPy `array_split` into five ordered bins. Fail closed unless bin row counts and area boundaries match Diversity-Axis X3.

For each bin report V-only pass@12, Mixed pass@12, and paired delta. The coverage-limited hypothesis is supported if and only if the smallest-bin delta is nonpositive. B3 and M1 deltas are included as secondary context but do not alter the hypothesis decision.
