# Amendment 001: same-budget method names

Date: 2026-08-16

Status: `FROZEN_BEFORE_SAME_BUDGET_IMPLEMENTATION_OR_RESULT`

The historical leave-UI-TARS value 63.88% is Allocation-Law's `M1_ccm`, not DECOMP's source-priority endpoint named `F1_majority`. Treating them as the same method would create another apparent contradiction.

The same-budget audit therefore reports three distinct aggregators for every pool:

1. `B3_mvp`: canonical ordered greedy complete-link B3;
2. `M1_ccm`: the historical fold-local CCM endpoint, required to reproduce the published 63.88% omit-UI-TARS value;
3. `source_priority`: the outer-development best-source endpoint used by DECOMP under the inherited `F1_majority` label.

The main historical reconciliation uses B3 and M1_ccm. Source-priority is an additional bridge to DECOMP/EVID terminology and is never substituted for M1_ccm.

No same-budget ICC result existed when this amendment was written.