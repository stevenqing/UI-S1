#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
RUN="runs/xfer/2026-08-07"

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 PID..." >&2
    exit 2
fi

tail_args=()
for pid in "$@"; do
    tail_args+=("--pid=$pid")
done
tail "${tail_args[@]}" -f /dev/null

test -d /proc/2274
.venv-scaleup/bin/python "$RUN/validate_stage1.py" --crop-set view1 --models CogAgent-18B

.venv-scaleup/bin/python "$RUN/infer/consensus_roi.py"
.venv-scaleup/bin/python "$RUN/validate_consensus.py"
.venv-scaleup/bin/python "$RUN/retention.py" backup \
    --source "$RUN/raw/mind2web-consensus-roi.jsonl" \
    --relative consensus/mind2web-consensus-roi.jsonl
.venv-scaleup/bin/python "$RUN/retention.py" verify

mkdir -p "$RUN/raw/stage2/cogagent" "$RUN/logs/stage2/cogagent"
pids=()
for shard in 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$shard" \
        runs/mind2web-cogagent/2026-07-28/run_python.sh \
        "$RUN/infer/crop_cogagent.py" \
        --regions "$RUN/raw/mind2web-consensus-roi.jsonl" \
        --sets C_uni,C_cond,C_rand,C_self \
        --output "$RUN/raw/stage2/cogagent/shard-$shard.jsonl" \
        --num-shards 8 --shard-index "$shard" --resume \
        >"$RUN/logs/stage2/cogagent/shard-$shard.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
test -d /proc/2274
if [[ "$failed" -ne 0 ]]; then
    exit "$failed"
fi

for shard in 4 5 6 7; do
    .venv-scaleup/bin/python "$RUN/retention.py" backup \
        --source "$RUN/raw/stage2/cogagent/shard-$shard.jsonl" \
        --relative "stage2/cogagent/shard-$shard.jsonl"
done
.venv-scaleup/bin/python "$RUN/retention.py" verify