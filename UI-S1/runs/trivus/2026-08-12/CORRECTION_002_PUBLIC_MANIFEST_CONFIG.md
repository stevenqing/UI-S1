# Correction 002: Public Manifest Config Scope

Date: 2026-08-12

Timing: after the corrected image-hash builder assembled and wrote one public JSONL, but before a public manifest, private labels, selector inference, or label-derived metric existed.

`build_public_records()` loaded and validated selector config internally, but `main()` referenced `config` while constructing `PUBLIC_MANIFEST.json` without binding it in its own scope. The process failed after writing the JSONL and before writing its manifest.

The unsealed 4,000-row file is retained as `invalid_public_attempt_002/public_records.unsealed.jsonl`, 3,406,130 bytes, SHA-256 `4e2de8d33ab45a0cd7acb33dad8db26dfbca35d8e71851cb8c3b8b7c99aaf7dd`. It is prohibited from inference and was moved out of the canonical data path.

`main()` now explicitly calls the same fail-closed `load_config()` before assembly. A regression test checks that every config-backed public-manifest field is present in the validated config. No bank content, image, candidate, prompt, model, or inference rule changes.