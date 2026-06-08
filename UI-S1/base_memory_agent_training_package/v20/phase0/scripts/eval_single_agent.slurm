#!/bin/bash
#SBATCH --job-name=eval_mem_1
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v20/phase0/scripts/logs/eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v20/phase0/scripts/logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate a single memory agent checkpoint
# Usage: sbatch --export=AGENT=minimal_next_step eval_single_agent.slurm
# ============================================

set -eo pipefail

if [ -z "$AGENT" ]; then
    echo "ERROR: AGENT not set. Use --export=AGENT=<name>"
    exit 1
fi

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

# Fix torch cache dir permission issue on some nodes
export TORCHINDUCTOR_CACHE_DIR="/scratch/a5l/shuqing.a5l/.cache/torch_inductor"
export TORCH_HOME="/scratch/a5l/shuqing.a5l/.cache/torch"
export HF_HOME="/scratch/a5l/shuqing.a5l/.cache/huggingface"
export TMPDIR="/scratch/a5l/shuqing.a5l/.cache/tmp"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TMPDIR"

mkdir -p "$OUTPUT_BASE" "$PROJECT_DIR/v20/phase0/scripts/logs"

# Checkpoint mapping
declare -A CHECKPOINTS
CHECKPOINTS[type_recovery]="$PROJECT_DIR/v20/phase0/output/full_proposal_type_recovery/checkpoint-50"
CHECKPOINTS[click_recovery]="$PROJECT_DIR/v20/phase0/output/full_proposal_click_recovery/checkpoint-50"
CHECKPOINTS[swipe_navigation]="$PROJECT_DIR/v20/phase0/output/full_proposal_swipe_navigation/checkpoint-50"
CHECKPOINTS[minimal_next_step]="$PROJECT_DIR/v20/phase0/output/full_proposal_minimal_next_step/checkpoint-75"
CHECKPOINTS[escape_finish]="$PROJECT_DIR/v20/phase0/output/full_proposal_escape_finish/checkpoint-75"
CHECKPOINTS[spreadsheet_formula]="$PROJECT_DIR/v20/phase0/output/full_proposal_spreadsheet_formula/checkpoint-75"

CKPT="${CHECKPOINTS[$AGENT]}"
OUT_DIR="$OUTPUT_BASE/$AGENT"

echo "=========================================="
echo "Single Agent Evaluation: $AGENT"
echo "Checkpoint: $CKPT"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "=========================================="

if [ -f "$OUT_DIR/summary.json" ]; then
    echo "SKIP: already evaluated"
    exit 0
fi

if [ ! -f "$CKPT/config.json" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

# Start vLLM server
echo "Starting vLLM server..."
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
    echo "ERROR: vLLM server failed to start after 300s"
    pkill -f "vllm" 2>/dev/null || true
    exit 1
fi

# Run evaluation with 32 threads (lower to avoid vLLM hangs)
echo "Running evaluation..."
python3 "$EVAL_SCRIPT" \
    --agent_type "$AGENT" \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:$PORT/v1" \
    --model_name memory_agent \
    --output_dir "$OUT_DIR" \
    --threads 32

EVAL_EXIT=$?
echo "Evaluation exit code: $EVAL_EXIT"

# Cleanup
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm" 2>/dev/null || true
sleep 5
pkill -9 -f "vllm" 2>/dev/null || true

echo "$AGENT done at $(date)"

if [ -f "$OUT_DIR/summary.json" ]; then
    cat "$OUT_DIR/summary.json"
fi
