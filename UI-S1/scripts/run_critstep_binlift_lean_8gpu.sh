#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/aiscuser/UI-S1/UI-S1}
cd "$PROJECT_DIR"

PYTHON=${PYTHON:-$PROJECT_DIR/.venv-qwen3-vllm/bin/python}
OUT=${OUT:-outputs/critstep_binlift_lean}
LOG_DIR=${LOG_DIR:-$OUT/logs}
mkdir -p "$LOG_DIR"

MODEL_PATH=${MODEL_PATH:-checkpoints/gui360-fullparam-sft-step250}
SERVED_MODEL=${SERVED_MODEL:-gui360-fullparam-sft-step250}
TEST_DATA=${TEST_DATA:-outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-outputs/gui360_history_ab/original_sft_template_gt_history_merged_20260630/summary.json}
REUSE_TEST_CANDIDATES=${REUSE_TEST_CANDIDATES:-outputs/verifier_e2e/slice200/candidates/per_step.jsonl}
TEST_CAND_DIR=${TEST_CAND_DIR:-$OUT/test_candidates}
TEST_CANDIDATES=${TEST_CANDIDATES:-$TEST_CAND_DIR/per_step.jsonl}
FILTER_SCORE_SET=${FILTER_SCORE_SET:-1}
FILTERED_TEST_CANDIDATES=${FILTERED_TEST_CANDIDATES:-$TEST_CAND_DIR/per_step_score_set.jsonl}
N_CANDIDATES=${N_CANDIDATES:-50}
EXPECTED_TEST_STEPS=${EXPECTED_TEST_STEPS:-7498}
SAMPLE_THREADS=${SAMPLE_THREADS:-64}
SAMPLES_PER_REQUEST=${SAMPLES_PER_REQUEST:-5}
COLLECT_LOGPROBS=${COLLECT_LOGPROBS:-0}
PORT0=${PORT0:-8141}
PORT1=${PORT1:-8142}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.65}
VLLM_GPU_MEMORY_UTILIZATION0=${VLLM_GPU_MEMORY_UTILIZATION0:-$VLLM_GPU_MEMORY_UTILIZATION}
VLLM_GPU_MEMORY_UTILIZATION1=${VLLM_GPU_MEMORY_UTILIZATION1:-$VLLM_GPU_MEMORY_UTILIZATION}
START_SERVER0=${START_SERVER0:-1}
START_SERVER1=${START_SERVER1:-1}
ALLOW_SERVER0_FALLBACK=${ALLOW_SERVER0_FALLBACK:-1}
GPU_STABLE_DELTA_MB=${GPU_STABLE_DELTA_MB:-512}
GPU_STABLE_CHECKS=${GPU_STABLE_CHECKS:-3}
GPU_STABLE_SLEEP=${GPU_STABLE_SLEEP:-5}

STAGE1_BASE=${STAGE1_BASE:-outputs/critstep_verifier_v2/gui360_fullparam_sft_step250_trainview}
STAGE1_ADAPTER=${STAGE1_ADAPTER:-outputs/critstep_verifier_v2/stage1_genrm_cot_lora}
STAGE2_ADAPTER=${STAGE2_ADAPTER:-outputs/critstep_verifier_v2/stage2_comparative_lora}
VERIFIER_N_SHARDS=${VERIFIER_N_SHARDS:-8}
VERIFIER_ROOT=${VERIFIER_ROOT:-$OUT/verifier}
STAGE1_DIR=$VERIFIER_ROOT/n${N_CANDIDATES}/stage1
STAGE2_DIR=$VERIFIER_ROOT/n${N_CANDIDATES}/stage2
SCORE_PER_STEP=$TEST_CANDIDATES

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=${VLLM_USE_V1:-1}

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

stop_servers() {
  if [[ ${#PIDS[@]} -eq 0 ]]; then
    return
  fi
  echo "[server] stopping owned vLLM servers"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
}

gpu_used_sum_mb() {
  local devices_csv=$1
  "$PYTHON" - "$devices_csv" <<'PY'
import subprocess, sys
devices = {int(item) for item in sys.argv[1].split(',') if item != ''}
out = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'
], text=True)
total = 0
for line in out.strip().splitlines():
    idx, used = [part.strip() for part in line.split(',')]
    if int(idx) in devices:
        total += int(float(used))
print(total)
PY
}

wait_gpu_memory_stable() {
  local devices_csv=$1
  local label=$2
  local prev=""
  local stable=0
  echo "[gpu] waiting for stable memory on ${label} devices=${devices_csv}"
  for attempt in $(seq 1 60); do
    local current
    current=$(gpu_used_sum_mb "$devices_csv")
    if [[ -n "$prev" ]]; then
      local delta=$(( current > prev ? current - prev : prev - current ))
      if [[ "$delta" -le "$GPU_STABLE_DELTA_MB" ]]; then
        stable=$((stable + 1))
      else
        stable=0
      fi
      echo "[gpu] ${label} memory_used_mb=${current} delta=${delta} stable=${stable}/${GPU_STABLE_CHECKS}"
      if [[ "$stable" -ge "$GPU_STABLE_CHECKS" ]]; then
        return 0
      fi
    else
      echo "[gpu] ${label} memory_used_mb=${current}"
    fi
    prev="$current"
    sleep "$GPU_STABLE_SLEEP"
  done
  echo "[gpu] WARNING: ${label} memory did not stabilize; continuing" >&2
}

start_server() {
  local devices=$1
  local port=$2
  local log_file=$3
  local gpu_mem_util=$4
  if curl -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "[server] reusing existing server on ${port}"
    return
  fi
  echo "[server] starting vLLM devices=${devices} port=${port} gpu_memory_utilization=${gpu_mem_util}"
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --port "$port" \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization "$gpu_mem_util" \
    --limit-mm-per-prompt '{"image": 1}' \
    >"$log_file" 2>&1 &
  PIDS+=("$!")
}

wait_server() {
  local port=$1
  local name=$2
  for attempt in $(seq 1 180); do
    if curl -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[server] ${name} ready on ${port}"
      return
    fi
    sleep 5
  done
  echo "[server] ERROR: ${name} not ready on ${port}" >&2
  exit 1
}

try_wait_server() {
  local port=$1
  local name=$2
  for attempt in $(seq 1 180); do
    if curl -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[server] ${name} ready on ${port}"
      return 0
    fi
    sleep 5
  done
  echo "[server] ${name} not ready on ${port}" >&2
  return 1
}

prepare_candidates() {
  mkdir -p "$TEST_CAND_DIR"
  if [[ ! -s "$TEST_CANDIDATES" && -s "$REUSE_TEST_CANDIDATES" ]]; then
    echo "[reuse] seeding candidates from $REUSE_TEST_CANDIDATES"
    cp "$REUSE_TEST_CANDIDATES" "$TEST_CANDIDATES"
  fi
}

sample_missing_test_steps() {
  echo "[phase0] sampling missing TEST steps at N=${N_CANDIDATES}"
  local existing_rows=0
  if [[ -s "$TEST_CANDIDATES" ]]; then
    existing_rows=$(wc -l < "$TEST_CANDIDATES")
  fi
  if [[ "$existing_rows" -ge "$EXPECTED_TEST_STEPS" ]]; then
    echo "[phase0] candidate pool already complete: ${existing_rows}/${EXPECTED_TEST_STEPS}; skipping vLLM sampling"
    return
  fi
  local api_urls=()
  if [[ "$START_SERVER1" = "1" ]]; then
    start_server "4,5,6,7" "$PORT1" "$LOG_DIR/vllm_${PORT1}.log" "$VLLM_GPU_MEMORY_UTILIZATION1"
    wait_server "$PORT1" server1
    api_urls+=("http://127.0.0.1:${PORT1}/v1")
  fi
  if [[ "$START_SERVER0" = "1" ]]; then
    wait_gpu_memory_stable "0,1,2,3" server0
    start_server "0,1,2,3" "$PORT0" "$LOG_DIR/vllm_${PORT0}.log" "$VLLM_GPU_MEMORY_UTILIZATION0"
    if try_wait_server "$PORT0" server0; then
      api_urls+=("http://127.0.0.1:${PORT0}/v1")
    elif [[ "$ALLOW_SERVER0_FALLBACK" = "1" && ${#api_urls[@]} -gt 0 ]]; then
      echo "[server] server0 failed; falling back to ${api_urls[*]}"
      pkill -TERM -f "vllm.entrypoints.openai.api_server.*${PORT0}" 2>/dev/null || true
    else
      echo "[server] ERROR: server0 failed and fallback is disabled/unavailable" >&2
      exit 1
    fi
  fi
  if [[ ${#api_urls[@]} -eq 0 ]]; then
    echo "[server] ERROR: no sample servers enabled" >&2
    exit 1
  fi
  local api_url_csv
  api_url_csv=$(IFS=,; echo "${api_urls[*]}")
  local logprob_flag=()
  if [[ "$COLLECT_LOGPROBS" = "1" ]]; then
    logprob_flag=(--collect-logprobs)
  fi
  "$PYTHON" scripts/verifier_e2e_sample.py \
    --test-data "$TEST_DATA" \
    --baseline-summary "$BASELINE_SUMMARY" \
    --api-url "$api_url_csv" \
    --model-name "$SERVED_MODEL" \
    --output-dir "$TEST_CAND_DIR" \
    --max-episodes 1000 \
    --n-candidates "$N_CANDIDATES" \
    --samples-per-request "$SAMPLES_PER_REQUEST" \
    --threads "$SAMPLE_THREADS" \
    --resume-from "$TEST_CANDIDATES" \
    "${logprob_flag[@]}" \
    2>&1 | tee "$LOG_DIR/sample_missing_test.log"
  stop_servers
}

score_stage1() {
  SCORE_PER_STEP="$TEST_CANDIDATES"
  if [[ "$FILTER_SCORE_SET" = "1" ]]; then
    echo "[filter] building score set: greedy-wrong AND recoverable"
    "$PYTHON" scripts/filter_verifier_score_set.py \
      --input "$TEST_CANDIDATES" \
      --output "$FILTERED_TEST_CANDIDATES" \
      --summary "$TEST_CAND_DIR/score_set_summary.json" \
      --n-candidates "$N_CANDIDATES" \
      2>&1 | tee "$LOG_DIR/filter_score_set.log"
    SCORE_PER_STEP="$FILTERED_TEST_CANDIDATES"
    echo "[filter] resetting Stage1/Stage2 outputs for filtered score set"
    rm -rf "$STAGE1_DIR" "$STAGE2_DIR"
  fi
  mkdir -p "$STAGE1_DIR"
  echo "[phase1] Stage1 scoring N=${N_CANDIDATES} on 8 GPUs"
  for shard in $(seq 0 $((VERIFIER_N_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" scripts/score_critstep_verifier_v2_cot_voting.py \
      --base-model "$STAGE1_BASE" \
      --adapter "$STAGE1_ADAPTER" \
      --per-step "$SCORE_PER_STEP" \
      --output-dir "$STAGE1_DIR" \
      --vote-ks 8 \
      --vote-chunk 8 \
      --score-mode verdict_vote \
      --device cuda:0 \
      --num-shards "$VERIFIER_N_SHARDS" \
      --shard-index "$shard" \
      --resume \
      >"$LOG_DIR/stage1_shard_${shard}.log" 2>&1 &
  done
  wait
  "$PYTHON" scripts/score_critstep_verifier_v2_cot_voting.py \
    --per-step "$SCORE_PER_STEP" \
    --output-dir "$STAGE1_DIR" \
    --vote-ks 8 \
    --num-shards "$VERIFIER_N_SHARDS" \
    --merge-shards \
    2>&1 | tee "$LOG_DIR/stage1_merge.log"
}

score_stage2() {
  mkdir -p "$STAGE2_DIR"
  echo "[phase1] Stage2 tournament N=${N_CANDIDATES} on 8 GPUs"
  for shard in $(seq 0 $((VERIFIER_N_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" scripts/score_critstep_verifier_stage2_comparative.py \
      --base-model "$STAGE1_BASE" \
      --adapter "$STAGE2_ADAPTER" \
      --per-step "$STAGE1_DIR/stage1_per_step.jsonl" \
      --stage1-summary "$STAGE1_DIR/stage1_summary.json" \
      --output-dir "$STAGE2_DIR" \
      --device cuda:0 \
      --num-shards "$VERIFIER_N_SHARDS" \
      --shard-index "$shard" \
      --resume \
      >"$LOG_DIR/stage2_shard_${shard}.log" 2>&1 &
  done
  wait
  "$PYTHON" scripts/score_critstep_verifier_stage2_comparative.py \
    --per-step "$STAGE1_DIR/stage1_per_step.jsonl" \
    --stage1-summary "$STAGE1_DIR/stage1_summary.json" \
    --output-dir "$STAGE2_DIR" \
    --num-shards "$VERIFIER_N_SHARDS" \
    --merge-shards \
    2>&1 | tee "$LOG_DIR/stage2_merge.log"
}

run_binlift() {
  echo "[phase2] recomposition bin-lift sweep"
  "$PYTHON" scripts/critstep_binlift.py \
    --test-candidates "$TEST_CANDIDATES" \
    --test-verifier-root "$VERIFIER_ROOT" \
    --n-candidates "$N_CANDIDATES" \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG_DIR/binlift.log"
}

echo "[lean] start $(date -u)" | tee "$LOG_DIR/overnight.log"
prepare_candidates | tee -a "$LOG_DIR/overnight.log"
sample_missing_test_steps | tee -a "$LOG_DIR/overnight.log"
score_stage1 | tee -a "$LOG_DIR/overnight.log"
score_stage2 | tee -a "$LOG_DIR/overnight.log"
run_binlift | tee -a "$LOG_DIR/overnight.log"
echo "[lean] end $(date -u)" | tee -a "$LOG_DIR/overnight.log"