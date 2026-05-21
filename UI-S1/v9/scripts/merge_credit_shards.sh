#!/bin/bash
# Merge v9 credit_cache_shard{N}.jsonl files into a single credit_cache.jsonl.
#
# Usage:
#   bash v9/scripts/merge_credit_shards.sh [credit_dir]
#
# Default credit_dir is v9/credits/v6_5_ac_ep4. Sorts records by sample_idx
# and writes:
#     ${credit_dir}/credit_cache.jsonl
# plus a tiny summary file with shard sizes.
set -e

CREDIT_DIR="${1:-/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/credits/v6_5_ac_ep4}"
OUT="${CREDIT_DIR}/credit_cache.jsonl"

if [ ! -d "$CREDIT_DIR" ]; then
    echo "ERROR: directory not found: $CREDIT_DIR" >&2
    exit 1
fi

shards=("$CREDIT_DIR"/credit_cache_shard*.jsonl)
if [ ! -e "${shards[0]}" ]; then
    echo "ERROR: no credit_cache_shard*.jsonl files in $CREDIT_DIR" >&2
    exit 1
fi

echo "Merging ${#shards[@]} shard(s) into $OUT ..."
python - "$OUT" "${shards[@]}" <<'PY'
import json, os, sys
out_path = sys.argv[1]
shard_paths = sys.argv[2:]
records = {}  # sample_idx -> record (later wins)
sizes = {}
for p in shard_paths:
    n = 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            records[r["sample_idx"]] = r
            n += 1
    sizes[os.path.basename(p)] = n

with open(out_path, "w") as f:
    for k in sorted(records):
        f.write(json.dumps(records[k], ensure_ascii=False) + "\n")

# tiny summary
n_total = len(records)
n_has_coord = sum(1 for r in records.values() if r.get("has_coord"))
n_coord_ok = sum(1 for r in records.values() if r.get("coord_correct") is True)
n_act_ok = sum(1 for r in records.values() if r.get("action_correct") is True)
n_parse_fail = sum(1 for r in records.values() if not r.get("parse_ok", True))

summary = {
    "n_total": n_total,
    "n_has_coord": n_has_coord,
    "n_coord_correct": n_coord_ok,
    "n_action_correct": n_act_ok,
    "n_parse_fail": n_parse_fail,
    "shards": sizes,
}
with open(out_path + ".summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
PY

echo "Wrote $OUT"
echo "Summary at ${OUT}.summary.json"
