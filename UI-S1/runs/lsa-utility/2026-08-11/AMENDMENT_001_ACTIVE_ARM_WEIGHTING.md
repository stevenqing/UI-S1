# Amendment 001 — Active-Arm Row Weighting

Date: 2026-08-11

Status: `PRE_RESULT`

The frozen specification simultaneously required constant-utility groups to be excluded and each underlying row to have fixed total weight with four arms receiving equal shares. When some arms are constant-utility, literal quarter weights leave different total active mass across rows.

Before any Utility-LSA result was observed, the executable rule is clarified:

1. Exclude constant-utility arm-groups because their group-relative advantage is undefined/non-informative.
2. Give every underlying benchmark row with at least one active arm the same total training mass.
3. Split that row mass equally across its active arms.
4. Split each active-arm mass equally across its 12 candidates.
5. Normalize each benchmark to equal total mass.

This prevents repeated arms/candidates from increasing a row's influence while preserving GTA-style exclusion of zero-variance groups. No reward, feature, model, threshold, or gate changes.
