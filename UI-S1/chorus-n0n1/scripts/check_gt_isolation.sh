#!/usr/bin/env bash
set -euo pipefail

if grep -R -n -E 'from +src\.bench\.scoring|import +src\.bench\.scoring|from +bench\.scoring|import +bench\.scoring' chorus-n0n1/src/readers chorus-n0n1/src/metrics/disagreement* 2>/dev/null; then
  echo 'GT-isolation check failed: scoring module import found in reader/disagreement code.' >&2
  exit 1
fi