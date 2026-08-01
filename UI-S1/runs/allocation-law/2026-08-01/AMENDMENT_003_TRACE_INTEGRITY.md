# Amendment 003: Production Trace Integrity

Date: 2026-08-01

Status: frozen after production generation completed and before any production JSONL row was loaded, scored, or aggregated.

The evaluation loader must recompute and verify `prediction_sha256` for both existing views 0-3 and extended views 4-11. It must also require each row's `stable_index` to equal the frozen N12 manifest index, `num_shards` to equal 4, and `shard_index` to equal `stable_index % 4`.

Extended traces must not contain `target_bbox`. Ground truth remains joined from the GTA1 source only after trace integrity and candidate geometry pass validation. Any mismatch fails closed before L1 or L2 evaluation.