# Amendment 001: Exact Recovery Execution

Date: 2026-08-12

Timing: after the result-free TriVUS preregistration and before copying a seed shard or starting recovery inference.

The already complete UI-AGILE Low/High two-shard lanes are added to the frozen recovery config with per-file SHA-256, bytes, row counts, shard indices, model revision, and model-index hash. They are never copied or regenerated.

The four incomplete single-shard files are copied byte-identically to the TriVUS recovery directory before resume. The historical files remain immutable. Each child command is exactly:

```text
.venv-ac-vllm/bin/python runs/xfer/2026-08-07/infer/ac_stage1_vllm.py \
  --model-id <frozen> --setting <low|high> --output <new-copy> \
  --num-shards 1 --shard-index 0 --batch-size 8 --resume
```

The environment-only GPU map is GUI-R1 Low/High on physical GPUs 1/3 and UI-R1-E Low/High on GPUs 5/7. No process is signaled, paused, reprioritized, or moved. Protected PID 2274 must remain present before and after recovery.

Protected-process checks bind PID 2274 to its frozen Linux start ticks, comm, command-line SHA-256, and executable path, preventing a recycled PID from satisfying the guard.

Reference JSONL rows contain historical `gt_*` fields, so the files are parsed, but recovery and validation code never indexes those keys. It reads only identity, image, instruction/history, and provenance fields. It never imports a scorer/evaluator or computes candidate success, accuracy, oracle coverage, or aggregation. Full rows must have unique continuous stable indices and appear in exact reference order for single-shard lanes; the complete two-shard UI-AGILE lanes must reconstruct exact reference order after sorting by `stable_index`.