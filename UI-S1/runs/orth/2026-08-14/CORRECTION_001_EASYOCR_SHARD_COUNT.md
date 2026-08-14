# ORTH Correction 001: EasyOCR shard count

Date: 2026-08-14
Timing: after RapidOCR completed, before any EasyOCR formal JSONL or lane manifest was published and before any Arm 1 matching statistic.

The 12-shard EasyOCR attempt processed 799/1,581 rows in roughly 52 minutes while leaving most node CPU cores idle. It was stopped before atomic rename, so it produced no formal EasyOCR JSONL or manifest. All 12 partial files and hashes are retained under `failed_attempts/easyocr_12shard/` and summarized by `FAILED_OCR_ATTEMPT_001.json`.

EasyOCR is rerun from row zero with 48 deterministic interleaved shards. RapidOCR remains the already completed 12-shard run. Engine versions, model hashes, OCR parameters, CPU-only execution, stable row order, raw schema, per-row fsync, and all downstream matching rules are unchanged. This correction changes only work partitioning and expected lane count.
