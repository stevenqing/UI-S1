#!/bin/bash
#SBATCH --job-name=eval_gnd_ckpt300
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_gnd_ckpt300_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_gnd_ckpt300_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate Grounding SFT v3 - ckpt300
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/grounding_sft_v3_ckpt300"

MODEL_NAME="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding/checkpoint-300"
MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=20110
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "=========================================="
echo "Grounding SFT v3 - ckpt300 Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    exit 1
fi

if [ ! -f "$MODEL_NAME/model.safetensors.index.json" ]; then
    echo "ERROR: Model checkpoint incomplete at $MODEL_NAME"
    exit 1
fi

echo "Dataset verified: $DATASET_ROOT"
echo "Model checkpoint verified: $MODEL_NAME"

echo "=========================================="
echo "Starting vLLM Server (port $VLLM_PORT)"
echo "=========================================="

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server     --model $MODEL_NAME     --port $VLLM_PORT     --tensor-parallel-size 4     --max-model-len 32768     --trust-remote-code     --gpu-memory-utilization 0.9     2>&1 | tee $LOG_DIR/vllm_gnd_ckpt300_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

cd $EVAL_DIR

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_eval() {
    local EVAL_TYPE=$1
    echo "=========================================="
    echo "Running $EVAL_TYPE evaluation"
    echo "=========================================="

    local OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR

    python evaluation.py         --root_dir $DATASET_ROOT         --type $EVAL_TYPE         --model_type $MODEL_TYPE         --model_name $MODEL_NAME         --api_url $API_URL         --threads $EVAL_THREADS         --output_dir $OUTPUT_DIR

    echo "$EVAL_TYPE completed. Results: $OUTPUT_DIR"
}

run_eval "grounding"
run_eval "action_prediction"
run_eval "action_prediction_a11y"

echo "=========================================="
echo "Stopping vLLM Server"
echo "=========================================="
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "Evaluation Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Results:"
ls -la $RESULTS_DIR/
echo ""
echo "Summaries:"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"
echo "=========================================="
