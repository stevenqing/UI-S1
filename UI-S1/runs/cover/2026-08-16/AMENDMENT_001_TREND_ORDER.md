# Amendment 001: source/stage trend ordering

Date: 2026-08-17

Status: `FROZEN_BEFORE_ARM_B_IMPLEMENTATION_OR_RESULT`

For each benchmark, order the three source/stage strata by descending mean phi. Exact numerical ties preserve this declared order:

1. `within_model_cross_slot`;
2. `cross_model_matched_role`;
3. `cross_model_unmatched_role`.

The two benchmarks are ordering-consistent only when their complete three-item ordering tuples are identical. This is descriptive; no trend fit, p-value, or threshold is introduced.