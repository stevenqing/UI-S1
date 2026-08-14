# OTEXT Correction 001: RapidOCR shard count

Date: 2026-08-14
Timing: before any RapidOCR formal JSONL/manifest and before Stage-0 statistics. EasyOCR continues unchanged.

The 12-shard RapidOCR run processed 524/1,581 rows while leaving substantial CPU capacity unused. Only those 12 OTEXT RapidOCR child processes were stopped. All partial files and hashes are retained under `failed_attempts/rapidocr_12shard/` and summarized by `FAILED_OCR_ATTEMPT_001.json`.

RapidOCR is rerun from row zero with 48 deterministic interleaved shards. Engine/version/model/config, one-thread CPU sessions, row order, raw schema, per-row fsync, and downstream analysis are unchanged. EasyOCR remains the concurrently running 96-shard primary run.
