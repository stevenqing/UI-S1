# CWIN retention recovery

Date: 2026-08-17

Status: `DECLARED_AFTER_SLOW_COPY_INTERRUPTION_BEFORE_ARCHIVE_RECOVERY`

The initial `retain_cwin.py` attempt copied each artifact independently to `/scratch/workspaceblobstore/cwin/2026-08-17/run`. The scratch backend imposed approximately one remote request latency per small file. After 293 files and more than eight minutes, the process was stopped to avoid an hour-scale metadata loop.

No source or backup file was deleted. The partial `run/` tree is retained as a failed attempt.

Recovery uses one uncompressed tar archive:

1. enumerate every run artifact except `STATUS.json` and `__pycache__` files;
2. compute and record each source file's byte size and SHA-256;
3. build a local tar containing each artifact under its run-relative path;
4. copy the tar once to the same scratch retention root;
5. reopen the scratch tar and verify every member's name, size, and SHA-256 against the source manifest;
6. write `BACKUP_MANIFEST.json` and local `STATUS.json` only after all member checks pass.

The recovery changes no CWIN geometry, statistic, gate, outcome, or authorization. The partial failed attempt remains preserved and is explicitly recorded in the manifest.