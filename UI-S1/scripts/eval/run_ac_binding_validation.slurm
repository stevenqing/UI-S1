#!/bin/bash
#SBATCH --job-name=ac_bind_val
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/ac_bind_val_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/ac_bind_val_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================================
# AC Binding Validation (Phase 1.5): Exp A/B/C
# ============================================================
# Runs eval_binding_validation_ac.py with vLLM server.
# Tests baseline/redbox/texthint/crop conditions on AC data.
#
# Requires: vLLM (ui-s1 conda env), 4 GPUs (TP=4)
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B}"
DATASET="$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl"
CONDITION="${CONDITION:-all}"
N_SAMPLES="${N_SAMPLES:-500}"
PORT=8000

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/evaluation/results/ac_binding_val_base_${TIMESTAMP}}"

echo "============================================================"
echo "AC Binding Validation: $CONDITION"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Condition: $CONDITION"
echo "Samples: $N_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/scripts/eval/logs"

# ── Start vLLM Server ──
echo "Starting vLLM server..."
vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port $PORT \
    --trust-remote-code \
    &
VLLM_PID=$!

# Wait for server
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for vLLM server... ($RETRY_COUNT/$MAX_RETRIES)"
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start within $((MAX_RETRIES * 10)) seconds"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
done
echo "vLLM server is ready!"

# ── Run Binding Validation ──
cd "$EVAL_DIR"

python -u eval_binding_validation_ac.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --condition "$CONDITION" \
    --n_samples "$N_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --vllm_port "$PORT"

echo "Binding validation complete at $(date)"

# ── Cleanup ──
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "Results: $OUTPUT_DIR"
cat "$OUTPUT_DIR/binding_validation_summary.json" 2>/dev/null || echo "(no summary yet)"
echo "============================================================"
