# Preregistration Amendment 006: P3 Allocation Details

Date: 2026-07-30

Status: applied while every P3 pool is still `PENDING_INFERENCE`, before any C1-C4 result can be generated.

## Reason

The original specification fixed the five-forward budget, four corners, grouped folds, and kappa objective, but did not define the first greedy unit, tie breaks, or how pending view units enter the kappa matrix. The W1 matrix contains only full-view model pairs and cannot score a `model/view` candidate without using W2 outcomes.

Using full-pool W2 failures would expose each test fold's labels to its allocation. This amendment closes that ambiguity with held-out selection.

## Fixed candidate pools

For each benchmark and setting, the allocation pool contains nine units:

- the fixed five C2 models at `full` view;
- `v1`, `v2`, `v3`, and `v4` for the fixed C1 representative, whose `full` unit is already in C2.

The C1 representative is GUI-R1-7B on AndroidControl and TongUI-7B on Mind2Web. The fixed C2 models remain those named in Amendment 003 and the original AndroidControl contract.

## Fold-local allocation

For each of the five frozen grouped folds:

1. compute every candidate unit's Step SR and binary failure vector on the other four folds;
2. initialize C3 with highest development Step SR, breaking ties lexicographically by unit key;
3. add the candidate with lowest mean Cohen failure kappa to the selected units;
4. break later ties by higher development Step SR, then lexicographically by unit key;
5. evaluate the selected five units only on the held-out fold.

C4 samples five distinct units without replacement from the same nine-unit pool. Its generator uses `SeedSequence([20260730, test_fold])`; PKA receives the sampled units in lexicographic order. C1 uses plurality followed by coordinate density medoid over the representative's five views. C2, C3, and C4 use uniform-weight joint PKA.

No test-fold label, success value, or kappa enters allocation.