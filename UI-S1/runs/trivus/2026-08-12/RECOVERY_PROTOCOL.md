# R0 AndroidControl Recovery Protocol

Date frozen: 2026-08-12

Timing: before copying partial shards or resuming any model inference.

R0 is a byte- and identity-audited continuation of the interrupted `runs/xfer/2026-08-07` stage-1 jobs. It does not modify historical files.

Execution:

1. verify source script, roster, reference manifests, model indices, and partial-shard SHA-256 values against `configs/recovery.yaml`;
2. copy each partial shard to `runs/trivus/2026-08-12/recovery/ac-stage1/.../shard-0.jsonl`;
3. verify copied bytes and completed IDs;
4. invoke the original script with the original `model-id`, setting, `num-shards=1`, `shard-index=0`, `batch-size=8`, and `--resume`;
5. preserve row-level flush and fsync;
6. require exactly 2,000 unique IDs matching the reference sample in every recovered lane;
7. verify row order, stable/source indices, episode, setting, source/image hashes, image size, model revision/index, prompt hashes, prediction schema, and shard identity;
8. copy the already complete UI-AGILE lanes into the final manifest by reference and verify their 2,000-row identities;
9. write and independently verify a per-file SHA-256 manifest.

No scorer, evaluator, label join, candidate oracle, aggregation, or accuracy function may be imported or called by R0 scripts. Failure preserves partial outputs and resumes from the last fsynced row.