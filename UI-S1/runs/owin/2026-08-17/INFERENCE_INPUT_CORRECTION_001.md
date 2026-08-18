# OWIN inference-input correction 001

Date: 2026-08-17

Status: `DECLARED_BEFORE_RUNNER_OR_GPU_AUTHORIZATION`

The first uncommitted inference-input build omitted target bbox coordinates and correctness but copied two evaluation-side window audit booleans: `target_center_contained` and `target_bbox_contained`. No model was loaded, no GPU was used, and no inference occurred.

That build is retained under `failed_attempts/inference_inputs_with_containment_fields/` and is prohibited from runner use. The corrected builder removes both fields from every formal and smoke window. Final inference inputs may contain only sample identity, execution shard, instruction, image path/hash, and frozen geometry fields.

This correction changes no sample, window coordinates, radius, call count, endpoint, or authorization. GPU remains unauthorized.