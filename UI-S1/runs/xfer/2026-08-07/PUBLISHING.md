# Published Experiment Data

This directory publishes the compact results, frozen protocols, evaluation code, and selected row-level traces required to recompute the completed Mind2Web Q1 transfer result.

Published row-level data includes:

- all three model full-image and view-1 stage-1 lanes;
- all three model four-arm stage-2 lanes;
- the TongUI views 2-16 bank used for MDE and budget curves;
- the target-free consensus RoI manifest;
- Mind2Web and AndroidControl proposer-ablation traces.

`PUBLICATION_MANIFEST.json` records the byte size, row count, and SHA-256 digest of every published trace. The compact primary outputs are `xf_mind2web.json`, `mde_mind2web.json`, `baseline_mind2web.json`, `STATUS.json`, `CONSOLIDATED_SUMMARY_ZH.md`, and `MIND2WEB_SCREENSPOT_CROSS_VALIDATION_ZH.md`.

Model weights, benchmark images, downloaded archives, source parquet files, and incomplete AndroidControl formal lanes are intentionally excluded. Complete retained traces are also independently hash-verified under the blobfuse path recorded in the publication manifest.