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
.venv-scaleup/bin/python "$RUN/validate_stage2.py" \
    --stage2-root "$RUN/raw/stage2" --models CogAgent-18B
for shard in 0 1 2 3 4 5 6 7; do
    .venv-scaleup/bin/python "$RUN/retention.py" backup \
        --source "$RUN/raw/stage2/cogagent/shard-$shard.jsonl" \
        --relative "stage2/cogagent/shard-$shard.jsonl"
done
.venv-scaleup/bin/python "$RUN/retention.py" verify

run_qwen_stage2() {
    local model_type="$1"
    local model_id="$2"
    local model_dir="$3"
    local output_name="$4"
    mkdir -p "$RUN/raw/stage2/$output_name" "$RUN/logs/stage2/$output_name"
    local pids=()
    for shard in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$shard" env -u PYTHONPATH \
            runs/mind2web-tongui/2026-07-28/.venv/bin/python \
            "$RUN/infer/crop_qwen.py" \
            --model-type "$model_type" --model-id "$model_id" --model-dir "$model_dir" \
            --regions "$RUN/raw/mind2web-consensus-roi.jsonl" \
            --sets C_uni,C_cond,C_rand,C_self \
            --output "$RUN/raw/stage2/$output_name/shard-$shard.jsonl" \
            --num-shards 8 --shard-index "$shard" --resume \
            >"$RUN/logs/stage2/$output_name/shard-$shard.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    test -d /proc/2274
    if [[ "$failed" -ne 0 ]]; then
        return "$failed"
    fi
    .venv-scaleup/bin/python "$RUN/validate_stage2.py" \
        --stage2-root "$RUN/raw/stage2" --models "$model_id"
    for shard in 0 1 2 3 4 5 6 7; do
        .venv-scaleup/bin/python "$RUN/retention.py" backup \
            --source "$RUN/raw/stage2/$output_name/shard-$shard.jsonl" \
            --relative "stage2/$output_name/shard-$shard.jsonl"
    done
    .venv-scaleup/bin/python "$RUN/retention.py" verify
}

run_qwen_stage2 \
    tongui TongUI-7B "$RUN/models/TongUI-7B" tongui
run_qwen_stage2 \
    uitars UI-TARS-7B "runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT" uitars