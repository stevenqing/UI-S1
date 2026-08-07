# Historical Cross-Benchmark Trace Loss Inventory

Date: 2026-08-07

## Confirmed missing derived artifact

- `runs/complementarity/2026-07-30/rows.parquet`
- Expected rows: 102,054
- Frozen expected SHA-256: `f3effb37b6979ceb8073bf7b47c80448ec9084da8991f3f93205b79f4f77c77e`
- Status: absent after workspace-wide search.

## Confirmed missing AndroidControl row traces

For both `low` and `high`, the historical `predictions.jsonl` files are absent for:

- UI-AGILE-3B;
- UI-AGILE-7B;
- UI-R1-E-3B;
- GUI-R1-3B;
- GUI-R1-7B.

The lane directories retain only aggregate `score.json` and `audit.json`. Aggregate files cannot reconstruct joint correctness, candidate coordinates, failure kappa, mixed-pool outputs, or paired transfer statistics.

## Confirmed missing Mind2Web row traces

Historical merged/full `predictions.jsonl` files are absent for the visual lanes used by the complementarity study, including:

- TongUI-3B/7B/32B;
- CogAgent-18B;
- UI-TARS-2B/7B/72B;
- ShowUI-2B;
- Qwen2.5-VL-3B/7B;
- SeeClick-9.6B derived row set.

Only aggregate score/audit artifacts and manifests remain.

## Consequence

The old D2 transfer experiment is not recomputable from retained artifacts. The 2026-08-07 transfer run therefore regenerates only the minimum Q1 roster and must keep all new row-level traces under both the local protected raw directory and the independent blobfuse backup root.

Old aggregate values remain historical context and are never combined with new row-level outputs in paired differences.
