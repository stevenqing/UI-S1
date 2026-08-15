# Amendment 002: Arm 1 contamination anchors and baselines

Date: 2026-08-15

Status: `FROZEN_BEFORE_ARM1_IMPLEMENTATION_OR_RESULT`

The five leaked ScreenSpot-Pro cells are disclosed fold-level outcome values in SPLIT, not row IDs. They are:

`[0.6388361796331435, 0.6388361796331435, 0.6306135357368754, 0.6255534471853258, 0.6325110689437066]`.

Arm 1 must not import SPLIT configuration or compare an inner-validation score to these values. Cell selection is reconstructed solely from row-level candidate correctness on the allowed inner split. The values are retained only in this contamination note.

The mandatory `dev_selection` baseline uses the frozen ScreenSpot method set from CLOSE/CEV:

`[majority, A0, ours, A1, A2, A3, A4]`.

It uses the same inner-train/inner-validation/outer-refit split as Arm 1. `majority` and `best_single` coincide for the all-POINT ScreenSpot pool under the frozen source-priority semantics; both names are reported to make that identity explicit. The existing V-only N4/N8/N12 values are historical fixed-budget comparators, not reconstructed C-uni subsets.

No Arm 1 matrix or result existed when this amendment was written.