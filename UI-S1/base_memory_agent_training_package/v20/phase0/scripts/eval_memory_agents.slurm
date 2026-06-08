#!/bin/bash
#SBATCH --job-name=eval_mem_agents
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v20/phase0/scripts/logs/eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v20/phase0/scripts/logs/eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate 6 memory agent checkpoints sequentially
# Each: vLLM serve (TP=4) -> eval with GT history -> kill server
# ============================================

set -eo pipefail

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TEST_DATA="$PROJECT_DIR/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
EVAL_SCRIPT="$PROJECT_DIR/v20/phase0/eval_memory_agents.py"
OUTPUT_BASE="$PROJECT_DIR/v20/phase0/eval_results"
PORT=8000

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

mkdir -p "$OUTPUT_BASE" "$PROJECT_DIR/v20/phase0/scripts/logs"

echo "=========================================="
echo "Memory Agent Evaluation (GT History)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start:  $(date)"
echo "=========================================="

# Agent -> best checkpoint mapping
declare -A CHECKPOINTS
CHECKPOINTS[type_recovery]="$PROJECT_DIR/v20/phase0/output/full_proposal_type_recovery/checkpoint-50"
CHECKPOINTS[click_recovery]="$PROJECT_DIR/v20/phase0/output/full_proposal_click_recovery/checkpoint-50"
CHECKPOINTS[swipe_navigation]="$PROJECT_DIR/v20/phase0/output/full_proposal_swipe_navigation/checkpoint-50"
CHECKPOINTS[minimal_next_step]="$PROJECT_DIR/v20/phase0/output/full_proposal_minimal_next_step/checkpoint-75"
CHECKPOINTS[escape_finish]="$PROJECT_DIR/v20/phase0/output/full_proposal_escape_finish/checkpoint-75"
CHECKPOINTS[spreadsheet_formula]="$PROJECT_DIR/v20/phase0/output/full_proposal_spreadsheet_formula/checkpoint-75"

AGENTS="type_recovery click_recovery swipe_navigation minimal_next_step escape_finish spreadsheet_formula"

for AGENT in $AGENTS; do
    CKPT="${CHECKPOINTS[$AGENT]}"
    OUT_DIR="$OUTPUT_BASE/$AGENT"

    echo ""
    echo "=========================================="
    echo "Evaluating: $AGENT"
    echo "Checkpoint: $CKPT"
    echo "Time: $(date)"
    echo "=========================================="

    # Skip if already evaluated
    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "SKIP: $AGENT - already evaluated (summary.json exists)"
        continue
    fi

    # Check checkpoint exists
    if [ ! -f "$CKPT/config.json" ]; then
        echo "SKIP: $AGENT - checkpoint not found at $CKPT"
        continue
    fi

    # Start vLLM server
    echo "Starting vLLM server for $AGENT..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$CKPT" \
        --served-model-name memory_agent \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.70 \
        --max-model-len 4096 \
        --port $PORT \
        --trust-remote-code \
        --dtype bfloat16 &
    VLLM_PID=$!

    # Wait for server to be ready (just poll the endpoint, don't check PID -
    # vLLM 0.16 forks and the parent PID may exit while children serve)
    echo "Waiting for vLLM server..."
    SERVER_READY=0
    for attempt in $(seq 1 300); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "Server ready after ${attempt}s"
            SERVER_READY=1
            break
        fi
        sleep 1
    done

    if [ "$SERVER_READY" -eq 0 ]; then
        echo "ERROR: vLLM server failed to start for $AGENT after 300s"
        pkill -f "vllm.*$CKPT" 2>/dev/null || true
        sleep 10
        continue
    fi

    # Run evaluation
    echo "Running evaluation for $AGENT..."
    python3 "$EVAL_SCRIPT" \
        --agent_type "$AGENT" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:$PORT/v1" \
        --model_name memory_agent \
        --output_dir "$OUT_DIR" \
        --threads 64

    EVAL_EXIT=$?
    echo "$AGENT evaluation exit code: $EVAL_EXIT"

    # Kill all vLLM processes and wait for GPU memory release
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm" 2>/dev/null || true
    sleep 10
    # Double check - force kill remaining
    pkill -9 -f "vllm" 2>/dev/null || true
    sleep 20

    echo "$AGENT done at $(date)"
done

# Print summary
echo ""
echo "=========================================="
echo "All Evaluations Complete"
echo "=========================================="
for AGENT in $AGENTS; do
    SUMMARY="$OUTPUT_BASE/$AGENT/summary.json"
    if [ -f "$SUMMARY" ]; then
        TSR=$(python3 -c "import json; print(json.load(open('$SUMMARY'))['TSR'])")
        STEP_SR=$(python3 -c "import json; print(json.load(open('$SUMMARY'))['StepSR'])")
        echo "$AGENT: TSR=${TSR}% StepSR=${STEP_SR}%"
    else
        echo "$AGENT: NO RESULTS"
    fi
done
echo "End: $(date)"
