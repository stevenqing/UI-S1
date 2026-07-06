#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/aiscuser/UI-S1/UI-S1}
cd "$PROJECT_DIR"

PYTHON=${PYTHON:-$PROJECT_DIR/.venv-qwen3-vllm/bin/python}
MODEL_PATH=${MODEL_PATH:-checkpoints/gui360-fullparam-sft-step250}
SERVED_MODEL=${SERVED_MODEL:-gui360-fullparam-sft-step250}
TEST_DATA=${TEST_DATA:-outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-outputs/gui360_history_ab/original_sft_template_gt_history_merged_20260630/summary.json}
OUT_ROOT=${OUT_ROOT:-outputs/verifier_e2e}
SLICE_EPISODES=${SLICE_EPISODES:-200}
RUN_FULL=${RUN_FULL:-0}
FULL_EPISODES=${FULL_EPISODES:-1000}
N_CANDIDATES=${N_CANDIDATES:-50}
N_SWEEP=${N_SWEEP:-5,10,20,50}
SAMPLE_THREADS=${SAMPLE_THREADS:-64}
SAMPLES_PER_REQUEST=${SAMPLES_PER_REQUEST:-5}
COLLECT_LOGPROBS=${COLLECT_LOGPROBS:-1}
RUN_VERIFIER=${RUN_VERIFIER:-0}
VERIFIER_N_SHARDS=${VERIFIER_N_SHARDS:-8}
STAGE1_BASE=${STAGE1_BASE:-outputs/critstep_verifier_v2/gui360_fullparam_sft_step250_trainview}
STAGE1_ADAPTER=${STAGE1_ADAPTER:-outputs/critstep_verifier_v2/stage1_genrm_cot_lora}
STAGE2_ADAPTER=${STAGE2_ADAPTER:-outputs/critstep_verifier_v2/stage2_comparative_lora}
PORT0=${PORT0:-8141}
PORT1=${PORT1:-8142}
LOG_DIR=${LOG_DIR:-$OUT_ROOT/logs}
mkdir -p "$LOG_DIR"

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
  echo "[server] Stopping owned vLLM servers"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
}

start_sampling_servers() {
  start_server "0,1,2,3" "$PORT0" "$LOG_DIR/vllm_${PORT0}.log"
  start_server "4,5,6,7" "$PORT1" "$LOG_DIR/vllm_${PORT1}.log"
  wait_server "$PORT0" "server0"
  wait_server "$PORT1" "server1"
}

start_server() {
  local devices=$1
  local port=$2
  local log_file=$3
  if curl -s "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "[server] Reusing existing server on port ${port}"
    return
  fi
  echo "[server] Starting vLLM on GPUs ${devices}, port ${port}"
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --port "$port" \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
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

run_sampling_eval() {
  local label=$1
  local max_episodes=$2
  local out_dir="$OUT_ROOT/$label"
  local candidate_dir="$out_dir/candidates"
  mkdir -p "$candidate_dir"
  local logprob_flag=()
  if [[ "$COLLECT_LOGPROBS" = "1" ]]; then
    logprob_flag=(--collect-logprobs)
  fi
  echo "[sample] ${label}: max_episodes=${max_episodes} N=${N_CANDIDATES}"
  "$PYTHON" scripts/verifier_e2e_sample.py \
    --test-data "$TEST_DATA" \
    --baseline-summary "$BASELINE_SUMMARY" \
    --api-url "http://127.0.0.1:${PORT0}/v1,http://127.0.0.1:${PORT1}/v1" \
    --model-name "$SERVED_MODEL" \
    --output-dir "$candidate_dir" \
    --max-episodes "$max_episodes" \
    --n-candidates "$N_CANDIDATES" \
    --samples-per-request "$SAMPLES_PER_REQUEST" \
    --threads "$SAMPLE_THREADS" \
    --resume-from "$candidate_dir/per_step.jsonl" \
    "${logprob_flag[@]}" \
    2>&1 | tee "$LOG_DIR/${label}_sample.log"
  "$PYTHON" scripts/verifier_e2e_eval.py \
    --candidate-files "$candidate_dir/per_step.jsonl" \
    --output-dir "$out_dir" \
    --n-sweep "$N_SWEEP" \
    --verifier-root "$out_dir/verifier" \
    2>&1 | tee "$LOG_DIR/${label}_eval_noverifier.log"
}

run_verifier_for_label() {
  local label=$1
  local out_dir="$OUT_ROOT/$label"
  local candidate_file="$out_dir/candidates/per_step.jsonl"
  local verifier_root="$out_dir/verifier"
  mkdir -p "$verifier_root/full/stage1"
  echo "[verifier] Stage1 full candidate scoring for ${label}"
  for shard in $(seq 0 $((VERIFIER_N_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" scripts/score_critstep_verifier_v2_cot_voting.py \
      --base-model "$STAGE1_BASE" \
      --adapter "$STAGE1_ADAPTER" \
      --per-step "$candidate_file" \
      --output-dir "$verifier_root/full/stage1" \
      --vote-ks 8 \
      --vote-chunk 8 \
      --score-mode verdict_vote \
      --device cuda:0 \
      --num-shards "$VERIFIER_N_SHARDS" \
      --shard-index "$shard" \
      --resume \
      >"$LOG_DIR/${label}_stage1_shard_${shard}.log" 2>&1 &
  done
  wait
  "$PYTHON" scripts/score_critstep_verifier_v2_cot_voting.py \
    --per-step "$candidate_file" \
    --output-dir "$verifier_root/full/stage1" \
    --vote-ks 8 \
    --merge-shards \
    2>&1 | tee "$LOG_DIR/${label}_stage1_merge.log"

  IFS=',' read -ra N_VALUES <<< "$N_SWEEP"
  for n in "${N_VALUES[@]}"; do
    local n_dir="$verifier_root/n${n}"
    mkdir -p "$n_dir/stage1" "$n_dir/stage2"
    "$PYTHON" scripts/verifier_e2e_prefix_pool.py \
      --input "$verifier_root/full/stage1/stage1_per_step.jsonl" \
      --output "$n_dir/stage1/stage1_per_step.jsonl" \
      --n-candidates "$n" \
      --vote-ks 8
    echo "[verifier] Stage2 N=${n} for ${label}"
    for shard in $(seq 0 $((VERIFIER_N_SHARDS - 1))); do
      CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" scripts/score_critstep_verifier_stage2_comparative.py \
        --base-model "$STAGE1_BASE" \
        --adapter "$STAGE2_ADAPTER" \
        --per-step "$n_dir/stage1/stage1_per_step.jsonl" \
        --stage1-summary "$verifier_root/full/stage1/stage1_summary.json" \
        --output-dir "$n_dir/stage2" \
        --device cuda:0 \
        --num-shards "$VERIFIER_N_SHARDS" \
        --shard-index "$shard" \
        --resume \
        >"$LOG_DIR/${label}_stage2_n${n}_shard_${shard}.log" 2>&1 &
    done
    wait
    "$PYTHON" scripts/score_critstep_verifier_stage2_comparative.py \
      --per-step "$n_dir/stage1/stage1_per_step.jsonl" \
      --stage1-summary "$verifier_root/full/stage1/stage1_summary.json" \
      --output-dir "$n_dir/stage2" \
      --merge-shards \
      2>&1 | tee "$LOG_DIR/${label}_stage2_n${n}_merge.log"
  done
  "$PYTHON" scripts/verifier_e2e_eval.py \
    --candidate-files "$candidate_file" \
    --output-dir "$out_dir" \
    --n-sweep "$N_SWEEP" \
    --verifier-root "$verifier_root" \
    2>&1 | tee "$LOG_DIR/${label}_eval_with_verifier.log"
}

echo "[e2e] Start $(date)"
start_sampling_servers
run_sampling_eval "slice200" "$SLICE_EPISODES"
if [[ "$RUN_VERIFIER" = "1" ]]; then
  stop_servers
  run_verifier_for_label "slice200"
fi

if [[ "$RUN_FULL" = "1" ]]; then
  start_sampling_servers
  run_sampling_eval "full1000" "$FULL_EPISODES"
  if [[ "$RUN_VERIFIER" = "1" ]]; then
    stop_servers
    run_verifier_for_label "full1000"
  fi
fi

echo "[e2e] Done $(date)"