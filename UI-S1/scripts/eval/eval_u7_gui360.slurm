#!/bin/bash
#SBATCH --job-name=eval_u7_gui360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_u7_gui360_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_u7_gui360_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Exp U7: Actor-Verifier on GUI-360
# 2 vLLM servers: V2 (action/verifier/resample) + V3 (grounding)
# ============================================

set -e

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_ROOT/scripts/eval/logs"

# V2: Qwen2.5-VL-7B-Instruct (Actor + Verifier + Resample)
V2_MODEL_PATH="${V2_MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct}"
V2_MODEL_NAME="${V2_MODEL_NAME:-Qwen2.5-VL-7B-Instruct}"
V2_PORT=19816
V2_TP=2

# V3: Grounding SFT model
V3_MODEL_PATH="${V3_MODEL_PATH:-$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding}"
V3_MODEL_NAME="${V3_MODEL_NAME:-gui360_full_sft_v3_grounding}"
V3_PORT=19815
V3_TP=2

OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_u7_gui360/${V2_MODEL_NAME}"

# Resample parameters
K=${K:-5}
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_WORKERS=${MAX_WORKERS:-8}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-0}

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

echo "=========================================="
echo "Exp U7: Actor-Verifier on GUI-360"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "V2 Model: $V2_MODEL_PATH (TP=$V2_TP, port=$V2_PORT)"
echo "V3 Model: $V3_MODEL_PATH (TP=$V3_TP, port=$V3_PORT)"
echo "K=$K, temperature=$TEMPERATURE, max_workers=$MAX_WORKERS"
echo "=========================================="

# --- Start V2 server (Actor + Verifier + Resample) ---
echo "Starting V2 vLLM server on port $V2_PORT..."
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model $V2_MODEL_PATH \
    --served-model-name $V2_MODEL_NAME \
    --port $V2_PORT \
    --tensor-parallel-size $V2_TP \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_v2_u7_${SLURM_JOB_ID}.log &
V2_PID=$!

# --- Start V3 server (Grounding) ---
echo "Starting V3 vLLM server on port $V3_PORT..."
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model $V3_MODEL_PATH \
    --served-model-name $V3_MODEL_NAME \
    --port $V3_PORT \
    --tensor-parallel-size $V3_TP \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_v3_u7_${SLURM_JOB_ID}.log &
V3_PID=$!

# --- Wait for both servers ---
wait_for_server() {
    local PORT=$1
    local NAME=$2
    local MAX_WAIT=600
    local WAIT=0
    echo "Waiting for $NAME server (port $PORT)..."
    while [ $WAIT -lt $MAX_WAIT ]; do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "$NAME server is ready!"
            return 0
        fi
        sleep 5
        WAIT=$((WAIT + 5))
    done
    echo "ERROR: $NAME server failed to start within ${MAX_WAIT}s"
    return 1
}

wait_for_server $V2_PORT "V2" || { kill $V2_PID $V3_PID 2>/dev/null; exit 1; }
wait_for_server $V3_PORT "V3" || { kill $V2_PID $V3_PID 2>/dev/null; exit 1; }

echo "Both servers ready. Starting evaluation..."

# --- Run evaluation ---
cd $PROJECT_ROOT

python scripts/eval/eval_u7_actor_verifier_gui360.py \
    --v2_endpoint http://localhost:${V2_PORT}/v1 \
    --v3_endpoint http://localhost:${V3_PORT}/v1 \
    --v2_model $V2_MODEL_NAME \
    --v3_model $V3_MODEL_NAME \
    --K $K \
    --temperature $TEMPERATURE \
    --max_workers $MAX_WORKERS \
    --max_trajectories $MAX_TRAJECTORIES \
    --output_dir $OUTPUT_DIR

EVAL_EXIT=$?

# --- Cleanup ---
kill $V2_PID $V3_PID 2>/dev/null
wait $V2_PID $V3_PID 2>/dev/null

echo "=========================================="
echo "U7 Actor-Verifier completed at $(date). Exit: $EVAL_EXIT"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
exit $EVAL_EXIT
