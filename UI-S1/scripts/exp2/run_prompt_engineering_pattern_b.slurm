#!/bin/bash
#SBATCH --job-name=prompt_eng_b
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/prompt_eng_b_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/prompt_eng_b_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Prompt Engineering on Pattern B Subset
# Model: SFT v2
# Modes: progress, scene, intent
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval"
LOG_DIR="$PROJECT_DIR/scripts/exp2/logs"
RESULTS_DIR="$PROJECT_DIR/scripts/exp2/results/prompt_engineering"

MODEL_NAME="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
MODEL_TYPE="qwen2.5_vl_7b"

DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19806
VLLM_HOST="localhost"
EVAL_THREADS=20

echo "=========================================="
echo "Prompt Engineering on Pattern B Subset"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Threads: $EVAL_THREADS"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# Generate Pattern B IDs
echo "Generating Pattern B trajectory IDs..."
python $PROJECT_DIR/scripts/exp2/dump_pattern_b_ids.py
PATTERN_B_FILE="$PROJECT_DIR/scripts/exp2/pattern_b_ids.json"

if [ ! -f "$PATTERN_B_FILE" ]; then
    echo "ERROR: Pattern B IDs file not generated"
    exit 1
fi

echo "Pattern B IDs: $(python -c "import json; print(len(json.load(open('$PATTERN_B_FILE'))))")"

if [ ! -d "$MODEL_NAME" ]; then
    echo "ERROR: SFT v2 model not found at $MODEL_NAME"
    exit 1
fi

# Start vLLM Server
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    2>&1 | tee $LOG_DIR/vllm_prompt_eng_${SLURM_JOB_ID}.log &

VLLM_PID=$!

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
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd $EVAL_DIR

for PROMPT_MODE in progress scene intent; do
    OUTPUT_DIR="$RESULTS_DIR/prompt_eng_${PROMPT_MODE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR

    echo ""
    echo "=========================================="
    echo "Running prompt mode: $PROMPT_MODE"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="

    python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type action_prediction_ar_${PROMPT_MODE} \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --api_url $API_URL \
        --threads $EVAL_THREADS \
        --ar_no_stop_on_error \
        --trajectory_ids_file $PATTERN_B_FILE \
        --output_dir $OUTPUT_DIR

    echo "Done: $PROMPT_MODE"
    cp $OUTPUT_DIR/*summary*.json $RESULTS_DIR/ 2>/dev/null || true
done

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "=========================================="
echo "All prompt engineering evaluations complete"
echo "End: $(date)"
echo "Results: $RESULTS_DIR"
echo "=========================================="
