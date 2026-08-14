# ORTH Correction 002: Final EasyOCR shard count

Date: 2026-08-14
Timing: before any formal EasyOCR JSONL/manifest and before Arm 1 metrics.

The 48-shard rerun processed 471/1,581 rows in roughly 12 minutes while using only about 40 of 96 CPU cores. It was stopped before atomic rename. Its 48 partial files and hashes are retained under `failed_attempts/easyocr_48shard/` and summarized by `FAILED_OCR_ATTEMPT_002.json`.

The final EasyOCR run uses 96 deterministic interleaved shards, one CPU thread each. This is the last shard-count adjustment. OCR engine/version/model/parameters, row assignment rule, full 1,581-row coverage, raw schema, and downstream analysis remain unchanged. RapidOCR remains the already locked 12-shard run.
