# Correction 001: Extracted PNG Hash Semantics

Date: 2026-08-12

Timing: after the first public-builder invocation failed on its first row, before any public-bank file or manifest was written, before selector inference, and before any private-label file existed.

`low_sample.jsonl` and `high_sample.jsonl` store SHA-256 of the original compressed image bytes embedded in parquet. `prepare_ac_rows.py` decodes those bytes to RGB and re-encodes PNG files for inference. Therefore the source-byte hash cannot equal the extracted PNG byte hash even when decoded image content is correct.

A result-blind audit of all 4,000 extracted files found:

- 4,000/4,000 files exist, decode as RGB, and match the frozen reference dimensions;
- 0/4,000 extracted PNG byte hashes equal the original compressed source-byte hash, confirming systematic re-encoding rather than isolated drift;
- all 2,000 paired Low/High actual PNG hashes match;
- 1,988 unique actual PNG hashes, reflecting repeated screenshots across steps.

The public `image_sha256` field is corrected to mean SHA-256 of the actual extracted PNG file read by the selector. Original source-byte hashes remain locked in R0 provenance but are never exposed as selector inputs. The public manifest records both hash semantics and the systematic mismatch count. No model, prompt, candidate, fold, or statistical rule changes.