#!/bin/bash
#SBATCH --job-name=eval_v10v2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/eval_v10_v2_ep1_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/eval_v10_v2_ep1_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v10.1 Eval — epoch-1 checkpoint
#
# v10.1 = v10 + grounder format reward (50/50 format+downstream)
# Eval via vLLM Multi-LoRA on AC eval set
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CHECKPOINT="$PROJECT_DIR/v10/output/v10_grpo_ddp_4node_v2/epoch-1"
EVAL_DATA="$PROJECT_DIR/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="$PROJECT_DIR/v10/output/eval_v2_ep1_vllm"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
PORT=8000
MAX_WORKERS=128

GROUNDER_ADAPTER="$CHECKPOINT/grounder/grounder"
ACTOR_ADAPTER="$CHECKPOINT/actor/actor"

mkdir -p "$PROJECT_DIR/v10/scripts/logs"
mkdir -p "$OUTPUT_DIR"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "=========================================="
echo "v10.1 Eval �� epoch-1 (grounder format reward)"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Checkpoint: $CHECKPOINT"
echo "Grounder:   $GROUNDER_ADAPTER"
echo "Actor:      $ACTOR_ADAPTER"
echo "Eval data:  $EVAL_DATA"
echo "Output:     $OUTPUT_DIR"
echo "Workers:    $MAX_WORKERS"
echo "Start:      $(date)"
echo "=========================================="

cd "$PROJECT_DIR"

# Step 1: Start vLLM server with multi-LoRA
echo "Step 1: Starting vLLM server with multi-LoRA..."
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port $PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enable-lora \
    --lora-modules "grounder=$GROUNDER_ADAPTER" "actor=$ACTOR_ADAPTER" \
    --max-lora-rank 128 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
VLLM_PID=$!

# Wait for server to be ready
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${WAIT_COUNT}s."
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "Waiting for vLLM server... ($WAIT_COUNT/$MAX_WAIT seconds)"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Step 2: Run eval
echo "Step 2: Running evaluation..."
python v10/eval_v10_vllm.py \
    --eval_data "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --port $PORT \
    --grounder_model "grounder" \
    --actor_model "actor" \
    --max_workers "$MAX_WORKERS"

EVAL_EXIT=$?

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Done: $(date)  exit=$EVAL_EXIT"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
exit $EVAL_EXIT
