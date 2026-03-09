#!/bin/bash
#SBATCH --job-name=eval_q35_9b
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_q35_9b_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_q35_9b_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# Evaluate Qwen3.5-9B on GUI-360
# coordinate_system=normalized_1000, resize_factor=32 (auto-detected via "qwen3" in name)
set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/qwen35_9b"

MODEL_NAME="$PROJECT_DIR/checkpoints/Qwen3.5-9B"
MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19806
VLLM_HOST="localhost"
EVAL_THREADS=20

echo "=========================================="
echo "Evaluate: Qwen3.5-9B on GUI-360"
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | Start: $(date)"
echo "Model: $MODEL_NAME"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Upgrade vllm + transformers to support qwen3_5 architecture
echo "Upgrading vllm and transformers for Qwen3.5 support..."
pip install --upgrade vllm 2>&1 | tail -5
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

mkdir -p $LOG_DIR $RESULTS_DIR

# Verify model exists
if [ ! -f "$MODEL_NAME/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_NAME"
    exit 1
fi

[ ! -d "$DATASET_ROOT" ] && echo "ERROR: Dataset not found" && exit 1

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_q35_9b_${SLURM_JOB_ID}.log &

VLLM_PID=$!

MAX_WAIT=600; WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1 && echo "vLLM ready!" && break
    sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT s)"
done
[ $WAIT_COUNT -ge $MAX_WAIT ] && echo "ERROR: vLLM timeout" && kill $VLLM_PID 2>/dev/null && exit 1

cd $EVAL_DIR
API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_eval() {
    local EVAL_TYPE=$1
    local OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR
    echo "Running $EVAL_TYPE..."
    python evaluation.py \
        --root_dir $DATASET_ROOT --type $EVAL_TYPE \
        --model_type $MODEL_TYPE --model_name $MODEL_NAME \
        --api_url $API_URL --threads $EVAL_THREADS --output_dir $OUTPUT_DIR
    echo "$EVAL_TYPE done. Results: $OUTPUT_DIR"
}

run_eval "grounding"
run_eval "action_prediction"

kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "Completed at $(date). Results: $RESULTS_DIR"
ls -la $RESULTS_DIR/
