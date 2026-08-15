# XSCR-LABEL-ISOLATED: physically isolated same-screen label study

Date: 2026-08-15

Status: `PREREGISTERED_PROCESS_SPEC_NOT_AUTHORIZED_FOR_EXECUTION`

Evidence status: `POST_SELECTION_PROCESS_VALIDATION`.

## Boundary

ScreenSpot-Pro labels have already been used across prior rounds. Physical isolation now cannot create a confirmatory set or erase post-selection. This specification is authorized only as process hardening for a possible exploratory follow-up. Independent confirmation requires new untouched data.

The negative prior is frozen from DECOMP Arm 2:

- 1,581 rows and 1,551 byte-unique screens;
- 98.52% singleton screens;
- 96.65% of rows on singleton screens;
- collision rows: 2/1,581 at 7 px, 0/1,581 at 14 px, and 4/1,581 at 28 px.

The expected effect is therefore below the 0.70 pp MDE unless a method changes outcomes outside the observed collision surface. No result may omit this prior.

## Physical sharding prerequisite

Execution is prohibited until an offline custodian creates a new immutable label store with one file per `image_sha256`. The sharder may read the current monolithic labels once, before any method implementation, and must produce:

- `public_index.jsonl`: sample key, image SHA-256, shard-relative path, bytes, SHA-256; no labels;
- `labels/<prefix>/<image_sha256>.jsonl`: only rows for that screen;
- `SHARD_MANIFEST.json`: complete source hash, exact partition, per-shard hashes, and zero missing/duplicate sample keys.

The sharding process computes no metric and exposes no label value to method code.

## Isolation

Before method implementation, screen hashes are assigned to development and sealed evaluation partitions by a committed seed. The development loader receives an OS-level allowlist containing only development shard paths. It must fail any `open`, glob, directory listing, mmap, archive read, or subprocess request outside that allowlist.

The sealed paths and their parent directory are unavailable to the development process. Opening all shards and filtering rows afterward is prohibited.

An open-audit log records path hash, operation, caller, and allow/deny result without recording label contents. The method, hyperparameters, baselines, random controls, and implementation tests must be committed before a separate one-shot evaluator receives the sealed allowlist.

## Scope

Any future method must be soft, transductive, and ScreenSpot-Pro-only. It must compare against majority, nested dev-selection, independent B3, and matched random reassignment. Q3/Q4 repair and damage are reported together. Runtime features may not include labels, target boxes, correctness, `ui_type`, or DECOMP evaluation statistics.

The sealed evaluation remains post-selection internal evidence because the dataset was previously used. Passing cannot produce a method claim; failing closes the direction without retuning.

## Authorization

This document does not authorize sharding, label access, method implementation, or evaluation. A separate committed execution amendment must identify the custodian process, source-label hashes, destination root, seed, allowlist mechanism, and retention path.