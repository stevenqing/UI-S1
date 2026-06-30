#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-.venv-qwen3-vllm/bin/python}
CKPT_ROOT=${CKPT_ROOT:-train_GUI_360/llamafactory/output/gui360_gt_history_full_sft}
DATASET=${DATASET:-train_GUI_360/llamafactory/data/gui360_gt_history_val.json}
CHECKPOINTS=${CHECKPOINTS:-checkpoint-13 checkpoint-26 checkpoint-39 checkpoint-52}
PROBES=${PROBES:-v1 v2 v4 v3}
OUT_ROOT=${OUT_ROOT:-outputs/gui360_history_ab/checkpoint_evals_$(date +%Y%m%d_%H%M%S)}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8125}
TP=${TP:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-65536}
MM_MAX_PIXELS=${MM_MAX_PIXELS:-200704}
LIMIT_MM_IMAGES=${LIMIT_MM_IMAGES:-24}
MAX_TOKENS=${MAX_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.0}
COORD_TOL=${COORD_TOL:-20.0}
LIMIT=${LIMIT:--1}
WORKERS=${WORKERS:-16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

V3_DIR=${V3_DIR:-$(ls -td outputs/gui360_history_ab/v3_candidates_* 2>/dev/null | head -1 || true)}
V3_PAIRS=${V3_PAIRS:-${V3_DIR:+$V3_DIR/G_v3_pairs.json}}
V3_SHUFFLE_PAIRS=${V3_SHUFFLE_PAIRS:-${V3_DIR:+$V3_DIR/G_v3_pairs_shuffle.json}}

SERVER_PID=""

log() {
  printf '[gt-history-eval] %s\n' "$*"
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping vLLM pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_for_server() {
  local base_url=$1
  local log_file=$2
  "$PYTHON" - "$base_url" "$log_file" <<'PY'
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

base_url = sys.argv[1].rstrip('/')
log_file = Path(sys.argv[2])
deadline = time.time() + 900
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(base_url + '/models', timeout=5) as response:
            if response.status == 200:
                print('ready')
                raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        last_error = exc
    if log_file.exists():
        text = log_file.read_text(errors='ignore')[-12000:]
        fatal_markers = ('Traceback (most recent call last)', 'CUDA out of memory', 'Address already in use')
        if any(marker in text for marker in fatal_markers):
            print(text)
            raise SystemExit(3)
    time.sleep(5)
print(f'timed out waiting for {base_url}/models; last_error={last_error}')
if log_file.exists():
    print(log_file.read_text(errors='ignore')[-12000:])
raise SystemExit(2)
PY
}

start_server() {
  local ckpt=$1
  local ckpt_path="$CKPT_ROOT/$ckpt"
  local out_dir=$2
  local server_log="$out_dir/server.log"
  if [[ ! -d "$ckpt_path" ]]; then
    echo "missing checkpoint path: $ckpt_path" >&2
    exit 2
  fi
  log "starting vLLM for $ckpt on $HOST:$PORT"
  server_args=(
    -m vllm.entrypoints.openai.api_server
    --model "$ckpt_path"
    --served-model-name "$ckpt"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --dtype bfloat16
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --limit-mm-per-prompt "{\"image\": $LIMIT_MM_IMAGES}"
    --mm-processor-kwargs "{\"max_pixels\": $MM_MAX_PIXELS}"
    --enforce-eager
    --disable-log-requests
    --disable-log-stats
  )
  if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    server_args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" VLLM_USE_V1=1 \
    "$PYTHON" "${server_args[@]}" \
      >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "http://$HOST:$PORT/v1" "$server_log"
}

run_probe() {
  local ckpt=$1
  local probe=$2
  local out_dir=$3
  local rows_dir="$out_dir/rows"
  local summaries_dir="$out_dir/summaries"
  local extra=()
  mkdir -p "$rows_dir" "$summaries_dir"
  if [[ "$probe" == "v3" ]]; then
    if [[ -z "${V3_PAIRS:-}" || ! -f "$V3_PAIRS" ]]; then
      log "skipping v3 for $ckpt; missing V3_PAIRS=$V3_PAIRS"
      return 0
    fi
    extra+=(--pairs "$V3_PAIRS")
    if [[ -n "${V3_SHUFFLE_PAIRS:-}" && -f "$V3_SHUFFLE_PAIRS" ]]; then
      extra+=(--shuffle-pairs "$V3_SHUFFLE_PAIRS")
    fi
  fi
  log "running $probe for $ckpt"
  "$PYTHON" -m gui360_long_horizon.experiments.capstone_runtime \
    --dataset "$DATASET" \
    --probe "$probe" \
    --arm G \
    --history-format gt_history \
    --base-url "http://$HOST:$PORT/v1" \
    --model "$ckpt" \
    --out-rows "$rows_dir/G_${probe}.jsonl" \
    --out-summary "$summaries_dir/G_${probe}.json" \
    --limit "$LIMIT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --coord-tol "$COORD_TOL" \
    --workers "$WORKERS" \
    "${extra[@]}" \
    >"$out_dir/${probe}.log" 2>&1
}

aggregate_one() {
  local ckpt=$1
  local out_dir=$2
  log "aggregating $ckpt"
  "$PYTHON" -m gui360_long_horizon.analysis.capstone_report \
    --summary-dir "$out_dir/summaries" \
    --results-out "$out_dir/capstone_results.json" \
    --out "$out_dir/verdict.json" \
    >"$out_dir/aggregate.log" 2>&1 || {
      log "aggregation failed for $ckpt; see $out_dir/aggregate.log"
      return 0
    }
}

write_matrix() {
  "$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for ckpt_dir in sorted(root.glob('checkpoint-*'), key=lambda p: int(p.name.rsplit('-', 1)[1])):
    item = {'checkpoint': ckpt_dir.name}
    for probe in ('v1', 'v2', 'v3', 'v4'):
        path = ckpt_dir / 'summaries' / f'G_{probe}.json'
        if path.exists():
            item[probe] = json.loads(path.read_text())
    verdict_path = ckpt_dir / 'verdict.json'
    if verdict_path.exists():
        item['verdict'] = json.loads(verdict_path.read_text()).get('verdict', {})
    rows.append(item)

(root / 'checkpoint_eval_matrix.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
lines = ['# GUI-360 GT-History Checkpoint Eval Matrix', '']
lines.append('| checkpoint | v1 matched | v1 none | v1 delta | v2 clean | v2 injected | v2 delta | v3 near | v3 far | v3 gap | v4 none | v4 oracle | v4 delta | verdict |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
def pct(x):
    return '' if x is None else f'{100*float(x):.1f}%'
for row in rows:
    v1 = row.get('v1', {})
    v2 = row.get('v2', {})
    v3 = row.get('v3', {})
    v4 = row.get('v4', {})
    verdict = row.get('verdict', {}).get('label', '')
    lines.append('| ' + ' | '.join([
        row['checkpoint'],
        pct(v1.get('acc_matched')),
        pct(v1.get('acc_none')),
        pct(v1.get('matched_minus_none')),
        pct(v2.get('acc_clean')),
        pct(v2.get('acc_injected')),
        pct(v2.get('injected_minus_clean')),
        pct(v3.get('near_acc')),
        pct(v3.get('far_acc')),
        pct(v3.get('near_minus_far')),
        pct(v4.get('acc_none')),
        pct(v4.get('acc_oracle')),
        pct(v4.get('oracle_minus_none')),
        verdict,
    ]) + ' |')
(root / 'checkpoint_eval_matrix.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(root / 'checkpoint_eval_matrix.md')
PY
}

main() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "missing python executable: $PYTHON" >&2
    exit 2
  fi
  mkdir -p "$OUT_ROOT"
  log "out_root: $OUT_ROOT"
  log "dataset: $DATASET"
  log "checkpoints: $CHECKPOINTS"
  log "probes: $PROBES"
  log "v3_pairs: ${V3_PAIRS:-none}"
  for ckpt in $CHECKPOINTS; do
    local out_dir="$OUT_ROOT/$ckpt"
    mkdir -p "$out_dir"
    stop_server
    start_server "$ckpt" "$out_dir"
    for probe in $PROBES; do
      if [[ "$probe" == "v3" ]]; then
        if ! run_probe "$ckpt" v3 "$out_dir"; then
          log "v3 failed for $ckpt; continuing"
        fi
      else
        run_probe "$ckpt" "$probe" "$out_dir"
      fi
    done
    aggregate_one "$ckpt" "$out_dir"
    write_matrix
  done
  stop_server
  write_matrix
  log "done: $OUT_ROOT"
}

trap stop_server EXIT
main "$@"