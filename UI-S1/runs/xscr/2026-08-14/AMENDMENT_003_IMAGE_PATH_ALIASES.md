# Amendment 003: image-path aliases

Date: 2026-08-14

Status: `FROZEN_AFTER_SECOND_FAILED_PUBLIC_ONLY_SEAL_BEFORE_ANY_SEAL_OR_XSCR_STATISTIC`

The second public-only seal attempt stopped before writing any artifact because one byte-identical screen hash maps to multiple retained image paths. Screen identity remains the SHA-256 of image bytes, not the local path.

The snapshot verifier now accepts path aliases only when every referenced file exists, hashes to the public `image_sha256`, and has identical dimensions. The manifest stores the lexically first workspace-relative path as canonical provenance and records the number of distinct source paths. No label was opened and no Q1-Q4 statistic existed when this amendment was written.