#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv-qwen3-vllm/bin/python}"
RUN_NAME="${RUN_NAME:-overnight_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/gui360_long_horizon/overnight/$RUN_NAME}"
LOG_DIR="$OUT_DIR/logs"
PID_DIR="$OUT_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

CONFIG="${CONFIG:-gui360_long_horizon/configs/default.yaml}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
LIMIT_STRONG="${LIMIT_STRONG:-1000}"
LIMIT_TEST="${LIMIT_TEST:-1000}"
LIMIT_SUCCESS_DIVERGENCE="${LIMIT_SUCCESS_DIVERGENCE:-1000}"
LIMIT_FAIL_TRAJ="${LIMIT_FAIL_TRAJ:-300}"
ADOPT_EXISTING_PROJECT_VLLM="${ADOPT_EXISTING_PROJECT_VLLM:-0}"
RUN_STRONG="${RUN_STRONG:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TEXTMEM="${RUN_TEXTMEM:-0}"
RUN_TEXTDRIFT="${RUN_TEXTDRIFT:-0}"
RUN_PLAN="${RUN_PLAN:-0}"
RUN_PREFETCH="${RUN_PREFETCH:-1}"
RUN_SUCCESS_IMAGE_PREP="${RUN_SUCCESS_IMAGE_PREP:-1}"
RUN_FAIL_PREP="${RUN_FAIL_PREP:-1}"
RUN_RECOVERY="${RUN_RECOVERY:-1}"
RUN_DIVERGENCE="${RUN_DIVERGENCE:-1}"
PREFETCH_SPLITS="${PREFETCH_SPLITS:-test,fail}"
PREFETCH_LIMIT_PER_SPLIT="${PREFETCH_LIMIT_PER_SPLIT:-0}"
MAX_TOKENS="${MAX_TOKENS:-128}"
LIMIT_SUCCESS_IMAGES="${LIMIT_SUCCESS_IMAGES:-$LIMIT_STRONG}"
LIMIT_TEXTMEM="${LIMIT_TEXTMEM:-200}"
LIMIT_TEXTDRIFT="${LIMIT_TEXTDRIFT:-200}"
LIMIT_PLAN="${LIMIT_PLAN:-200}"
LIMIT_TEXTDRIFT_BASE="${LIMIT_TEXTDRIFT_BASE:-0}"
TEXTDRIFT_MAX_INJECTED="${TEXTDRIFT_MAX_INJECTED:-3}"
TEXTDRIFT_BASELINE_ROWS="${TEXTDRIFT_BASELINE_ROWS:-}"
TEXTMEM_GATE_EPS="${TEXTMEM_GATE_EPS:-0.01}"
TEXTMEM_HISTORY_MODES="${TEXTMEM_HISTORY_MODES:-full,summary,corrupt,none}"
SCORE_THREADS="${SCORE_THREADS:-32}"
export GUI360_FORCE_REMOTE_SHARDS="${GUI360_FORCE_REMOTE_SHARDS:-0}"

STRONG_MODEL="${STRONG_MODEL:-checkpoints/Qwen2.5-VL-72B-Instruct}"
TEST_MODEL="${TEST_MODEL:-checkpoints/gui360-fullparam-sft-step250}"
STRONG_PORT="${STRONG_PORT:-8001}"
TEST_PORT="${TEST_PORT:-8000}"
STRONG_KV_CACHE_MEMORY_BYTES="${STRONG_KV_CACHE_MEMORY_BYTES:-17179869184}"
STRONG_MAX_NUM_SEQS="${STRONG_MAX_NUM_SEQS:-64}"
STRONG_MAX_NUM_BATCHED_TOKENS="${STRONG_MAX_NUM_BATCHED_TOKENS:-16384}"
TEST_MAX_NUM_SEQS="${TEST_MAX_NUM_SEQS:-32}"
TEST_MAX_NUM_BATCHED_TOKENS="${TEST_MAX_NUM_BATCHED_TOKENS:-8192}"
TEST_KV_CACHE_MEMORY_BYTES="${TEST_KV_CACHE_MEMORY_BYTES:-8589934592}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_DIR/overnight.log"; }

port_ready() {
  local port="$1"
  "$PY" - "$port" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port=sys.argv[1]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
        json.loads(r.read().decode("utf-8"))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
}

find_project_vllm_pid() {
  local port="$1" model="$2"
  pgrep -af 'vllm.entrypoints.openai.api_server' | while read -r pid cmd; do
    if [[ "$cmd" == *"--port $port"* && "$cmd" == *"$model"* ]]; then
      local cwd
      cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
      if [[ "$cwd" == "$ROOT" ]]; then
        echo "$pid"
        return 0
      fi
    fi
  done | head -n 1
}

wait_for_port() {
  local port="$1" name="$2" max_wait="${3:-1800}"
  local start now
  start="$(date +%s)"
  while true; do
    if port_ready "$port"; then
      log "$name is ready on port $port"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start > max_wait )); then
      log "$name did not become ready on port $port within ${max_wait}s"
      return 1
    fi
    sleep 15
  done
}

start_or_reuse_strong() {
  local pidfile="$PID_DIR/strong_72b.pid"
  if port_ready "$STRONG_PORT"; then
    log "reusing existing strong server on $STRONG_PORT"
    if [[ "$ADOPT_EXISTING_PROJECT_VLLM" == "1" ]]; then
      local pid
      pid="$(find_project_vllm_pid "$STRONG_PORT" "$STRONG_MODEL" || true)"
      if [[ -n "$pid" ]]; then
        echo "$pid" > "$pidfile"
        log "adopted project strong vLLM pid=$pid for later stop"
      else
        log "existing strong server was not adopted; it will not be stopped"
      fi
    fi
    return 0
  fi
  log "starting strong 72B server on $STRONG_PORT using GPUs $GPU_DEVICES"
  NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES="$GPU_DEVICES" nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$STRONG_MODEL" \
    --served-model-name "$STRONG_MODEL" \
    --host 0.0.0.0 --port "$STRONG_PORT" \
    --dtype bfloat16 --tensor-parallel-size 8 \
    --max-model-len 4096 --gpu-memory-utilization 0.60 \
    --kv-cache-memory-bytes "$STRONG_KV_CACHE_MEMORY_BYTES" \
    --max-num-seqs "$STRONG_MAX_NUM_SEQS" --max-num-batched-tokens "$STRONG_MAX_NUM_BATCHED_TOKENS" \
    --skip-mm-profiling --enforce-eager \
    --limit-mm-per-prompt '{"image":1}' --disable-log-requests \
    > "$LOG_DIR/vllm_72b.log" 2>&1 &
  echo "$!" > "$pidfile"
  wait_for_port "$STRONG_PORT" "strong 72B" 2400
}

start_or_reuse_test() {
  local pidfile="$PID_DIR/test_sft.pid"
  if port_ready "$TEST_PORT"; then
    log "reusing existing test server on $TEST_PORT"
    if [[ "$ADOPT_EXISTING_PROJECT_VLLM" == "1" ]]; then
      local pid
      pid="$(find_project_vllm_pid "$TEST_PORT" "$TEST_MODEL" || true)"
      if [[ -n "$pid" ]]; then
        echo "$pid" > "$pidfile"
        log "adopted project test vLLM pid=$pid for later stop"
      else
        log "existing test server was not adopted; it will not be stopped"
      fi
    fi
    return 0
  fi
  if port_ready "$STRONG_PORT" && [[ ! -f "$PID_DIR/strong_72b.pid" ]]; then
    log "strong server is still running and not owned by this script; skipping test-model server to avoid touching external/project state"
    return 2
  fi
  log "starting test SFT server on $TEST_PORT using GPU ${TEST_GPU_DEVICES:-0}"
  CUDA_VISIBLE_DEVICES="${TEST_GPU_DEVICES:-0}" nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$TEST_MODEL" \
    --served-model-name "$TEST_MODEL" \
    --host 0.0.0.0 --port "$TEST_PORT" \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 \
    --kv-cache-memory-bytes "$TEST_KV_CACHE_MEMORY_BYTES" --skip-mm-profiling --enforce-eager \
    --max-num-seqs "$TEST_MAX_NUM_SEQS" --max-num-batched-tokens "$TEST_MAX_NUM_BATCHED_TOKENS" \
    --limit-mm-per-prompt '{"image":1}' --disable-log-requests \
    > "$LOG_DIR/vllm_sft.log" 2>&1 &
  echo "$!" > "$pidfile"
  wait_for_port "$TEST_PORT" "test SFT" 1200
}

stop_owned() {
  local name="$1"
  local pidfile="$PID_DIR/$name.pid"
  if [[ ! -f "$pidfile" ]]; then
    log "no owned pidfile for $name; not stopping anything"
    return 0
  fi
  local pid
  pid="$(cat "$pidfile")"
  if [[ -z "$pid" || ! -d "/proc/$pid" ]]; then
    log "owned $name pid is gone"
    return 0
  fi
  local cmd
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  if [[ "$cmd" != *"vllm.entrypoints.openai.api_server"* ]]; then
    log "pid $pid for $name is not vLLM; refusing to stop"
    return 0
  fi
  log "stopping owned $name pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    [[ ! -d "/proc/$pid" ]] && return 0
    sleep 2
  done
  log "owned $name pid=$pid still alive after SIGTERM; sending SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
}

phase() {
  log "phase: $*"
  "$@" 2>&1 | tee -a "$LOG_DIR/overnight.log"
}

cleanup() {
  stop_owned test_sft || true
  stop_owned strong_72b || true
}
trap cleanup EXIT

log "overnight run: $RUN_NAME"
log "out_dir: $OUT_DIR"
log "external GPU processes are not managed; only pidfiles under $PID_DIR can be stopped"

phase "$PY" -m gui360_long_horizon.overnight ensure-fail-images --config "$CONFIG" --out_dir "$OUT_DIR"
if [[ "$RUN_PREFETCH" == "1" ]]; then
  phase "$PY" -m gui360_long_horizon.overnight prefetch-jsonl --config "$CONFIG" --out_dir "$OUT_DIR" --prefetch_splits "$PREFETCH_SPLITS" --prefetch_limit_per_split "$PREFETCH_LIMIT_PER_SPLIT"
else
  log "skipping JSONL prefetch because RUN_PREFETCH=$RUN_PREFETCH"
fi
if [[ "$RUN_SUCCESS_IMAGE_PREP" == "1" ]]; then
  phase "$PY" -m gui360_long_horizon.overnight extract-success-images --config "$CONFIG" --out_dir "$OUT_DIR" --limit_success_images "$LIMIT_SUCCESS_IMAGES"
else
  log "skipping success image extraction because RUN_SUCCESS_IMAGE_PREP=$RUN_SUCCESS_IMAGE_PREP"
fi
if [[ "$RUN_FAIL_PREP" == "1" ]]; then
  phase "$PY" -m gui360_long_horizon.overnight extract-fail-images --config "$CONFIG" --out_dir "$OUT_DIR" --limit_fail_traj "$LIMIT_FAIL_TRAJ"
else
  log "skipping fail image extraction because RUN_FAIL_PREP=$RUN_FAIL_PREP"
fi
phase "$PY" -m gui360_long_horizon.overnight loader-smoke --config "$CONFIG" --out_dir "$OUT_DIR"

if [[ "$RUN_STRONG" == "1" ]]; then
  start_or_reuse_strong
  phase "$PY" -m gui360_long_horizon.overnight score-success \
    --config "$CONFIG" --out_dir "$OUT_DIR" \
    --api_url "http://localhost:$STRONG_PORT/v1" --model_name "$STRONG_MODEL" \
    --score_label strong_72b --limit_steps "$LIMIT_STRONG" --history_mode none --input_mode visual --max_tokens "$MAX_TOKENS" --threads "$SCORE_THREADS"
  stop_owned strong_72b
fi

if [[ "$RUN_TEST" == "1" ]]; then
  if start_or_reuse_test; then
    phase "$PY" -m gui360_long_horizon.overnight score-success \
      --config "$CONFIG" --out_dir "$OUT_DIR" \
      --api_url "http://localhost:$TEST_PORT/v1" --model_name "$TEST_MODEL" \
      --score_label test_sft --limit_steps "$LIMIT_TEST" --history_mode none --input_mode visual --max_tokens "$MAX_TOKENS" --threads "$SCORE_THREADS"
    stop_owned test_sft
  else
    log "test-model phase skipped because server could not be started safely"
  fi
fi

if [[ "$RUN_TEXTMEM" == "1" ]]; then
  if start_or_reuse_test; then
    phase "$PY" -m gui360_long_horizon.overnight textmem-gate \
      --config "$CONFIG" --out_dir "$OUT_DIR" \
      --api_url "http://localhost:$TEST_PORT/v1" --model_name "$TEST_MODEL" \
      --limit_steps "$LIMIT_TEXTMEM" --history_modes "$TEXTMEM_HISTORY_MODES" \
      --input_mode visual --max_tokens "$MAX_TOKENS" --threads "$SCORE_THREADS" --gate_eps "$TEXTMEM_GATE_EPS"
    stop_owned test_sft
  else
    log "textmem phase skipped because test server could not be started safely"
  fi
fi

if [[ "$RUN_TEXTDRIFT" == "1" ]]; then
  if start_or_reuse_test; then
    phase "$PY" -m gui360_long_horizon.overnight textdrift-gate \
      --config "$CONFIG" --out_dir "$OUT_DIR" \
      --api_url "http://localhost:$TEST_PORT/v1" --model_name "$TEST_MODEL" \
      --limit_steps "$LIMIT_TEXTDRIFT" --limit_textdrift_base "$LIMIT_TEXTDRIFT_BASE" \
      --limit_fail_traj "$LIMIT_FAIL_TRAJ" --max_injected "$TEXTDRIFT_MAX_INJECTED" \
      --baseline_rows "$TEXTDRIFT_BASELINE_ROWS" --input_mode visual \
      --max_tokens "$MAX_TOKENS" --threads "$SCORE_THREADS"
    stop_owned test_sft
  else
    log "textdrift phase skipped because test server could not be started safely"
  fi
fi

if [[ "$RUN_PLAN" == "1" ]]; then
  if start_or_reuse_test; then
    phase "$PY" -m gui360_long_horizon.overnight plan-gate \
      --config "$CONFIG" --out_dir "$OUT_DIR" \
      --api_url "http://localhost:$TEST_PORT/v1" --model_name "$TEST_MODEL" \
      --limit_steps "$LIMIT_PLAN" --history_mode none --input_mode visual \
      --max_tokens "$MAX_TOKENS" --threads "$SCORE_THREADS"
    stop_owned test_sft
  else
    log "plan phase skipped because test server could not be started safely"
  fi
fi

phase "$PY" -m gui360_long_horizon.overnight validity-report --config "$CONFIG" --out_dir "$OUT_DIR" --strong_label strong_72b --test_label test_sft
if [[ "$RUN_RECOVERY" == "1" ]]; then
  phase "$PY" -m gui360_long_horizon.overnight recovery-scan --config "$CONFIG" --out_dir "$OUT_DIR" --limit_fail_traj "$LIMIT_FAIL_TRAJ"
else
  log "skipping recovery scan because RUN_RECOVERY=$RUN_RECOVERY"
fi

if [[ "$RUN_DIVERGENCE" == "1" ]]; then
  phase "$PY" -m gui360_long_horizon.overnight divergence-scan \
    --config "$CONFIG" --out_dir "$OUT_DIR" \
    --limit_success_steps "$LIMIT_SUCCESS_DIVERGENCE" --limit_fail_traj "$LIMIT_FAIL_TRAJ"
fi

phase "$PY" -m gui360_long_horizon.overnight summary --config "$CONFIG" --out_dir "$OUT_DIR"
log "overnight complete"
