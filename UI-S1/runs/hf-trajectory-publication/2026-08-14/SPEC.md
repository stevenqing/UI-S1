# Hugging Face trajectory publication protocol

Date: 2026-08-14

Status: `PREREGISTERED_BEFORE_PACKAGE_ASSEMBLY`

## Purpose

Publish the retained label-blind Mind2Web and AndroidControl GUI model trajectories to two private Hugging Face dataset repositories. This is archival publication, not a new experiment and not a result-producing round.

## Destination repositories

- `Stevenshuqing/UI-S1-Mind2Web-Trajectories`
- `Stevenshuqing/UI-S1-AndroidControl-Trajectories`

Both repositories must be created as private datasets. Public visibility requires a later license and redistribution review.

## Canonical sources

Mind2Web uses the XFER retention manifest at `/scratch/workspaceblobstore/xfer-traces/2026-08-07/BACKUP_MANIFEST.json`. The package includes only original model-output shards corresponding to these local source families:

- `raw/stage1/`
- `raw/stage1/view1/`
- `raw/stage2/`
- `raw/views/`

AndroidControl uses two retained sources:

- `/scratch/workspaceblobstore/aggmatch-traces/2026-08-09/BACKUP_MANIFEST.json`: all `raw/ac-stage1/**/*.jsonl` model-output shards.
- `/scratch/workspaceblobstore/trivus/2026-08-12/BACKUP_MANIFEST.json`: all `recovery/ac-stage1/**/*.jsonl` model-output shards.

Files are deduplicated by SHA-256. Semantically overlapping partial and complete lanes remain separate when their bytes differ; provenance paths distinguish them.

## Mandatory exclusions

The publication must not include benchmark images, source archives or parquet files, model weights, ground-truth actions or coordinates, target boxes, candidate-success labels, private labels/scales, rewards, correctness fields, selector labels/logits, training features, derived benchmark statistics, invalid leakage attempts, credentials, tokens, or absolute local paths.

In particular, XFER proposer-ablation and consensus-RoI outputs are excluded because they contain evaluation-side target fields. TriVUS selector/formal outputs and all downstream statistical artifacts are excluded because they are not raw GUI model trajectories.

## Package and validation

The release builder must:

1. Resolve every source only through an authoritative retention manifest.
2. Verify source bytes and SHA-256 before copying.
3. Parse every nonblank JSONL row.
4. Reject forbidden keys recursively and reject absolute local path values.
5. Preserve raw JSONL bytes without rewriting records.
6. Write a per-file release manifest with source run, source SHA-256, bytes, rows, and destination path.
7. Add a dataset card that states the package is label-blind, excludes images and labels, is private pending license review, and must be joined to separately licensed benchmark data by stable IDs and image hashes.
8. Reverify every staged file against the release manifest before upload.

## Upload and remote verification

Each repository is created with `repo_type=dataset` and `private=true`. Uploads use one explicit Hub commit per repository. After upload, the publisher must list remote files, download the remote release manifest and dataset card, compare their SHA-256 to the local package, and verify every expected path is present. A publication status file records repository IDs, commit revisions, local and remote manifest hashes, file counts, bytes, and verification time.

No token value may be logged or stored. Authentication reporting is limited to the Hugging Face username.

## Failure policy

If validation fails, no upload occurs. If one repository upload succeeds and the other fails, retain the successful private repository, record the partial state, and resume only the missing repository without rewriting the successful commit. Never make either repository public automatically.