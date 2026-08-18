# OWIN evaluation serialization correction 001

Date: 2026-08-18

Status: `DECLARED_AFTER_COMPUTATION_BEFORE_ANY_ARM_A_JSON_RESULT`

The first complete CPU evaluation computed all endpoints and bootstrap diagnostics, then failed while atomically serializing `ARM_A.json`. `constant_slot_count` was a NumPy `int64`, which Python's standard JSON encoder rejects. No `ARM_A.json` was created and no result was read from the incomplete temporary file.

The incomplete `ARM_A.json.tmp` and the already-written 500-row private evaluation JSONL are retained under `failed_attempts/evaluation_serialization_001/`. Neither is used as the final artifact. The correction converts NumPy integer, floating, and boolean scalar containers to their Python scalar equivalents during JSON encoding and explicitly casts `constant_slot_count` to `int`.

This changes no trace, label join, candidate, fold, bootstrap sample, seed, statistic, threshold, or interpretation. The deterministic CPU evaluator is rerun from the same frozen inputs.